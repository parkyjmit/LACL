import dgl
import dgl.function as fn
import torch
from dgl.nn import AvgPooling

from model.utils import RBFExpansion
import torch.nn.functional as F
from torch import nn


class EGGConv(nn.Module):

    def __init__(
        self, input_features: int, output_features: int):
        """Initialize parameters for ALIGNN update."""
        super().__init__()
        self.src_gate = nn.Linear(input_features, output_features)
        self.dst_gate = nn.Linear(input_features, output_features)
        self.edge_gate = nn.Linear(input_features, output_features)
        self.src_update = nn.Linear(input_features, output_features)
        self.dst_update = nn.Linear(input_features, output_features)

    def forward(
        self,
        g: dgl.DGLGraph,
        node_feats: torch.Tensor,
        edge_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Edge-gated graph convolution.
        h_i^l+1 = ReLU(U h_i + sum_{j->i} eta_{ij} ⊙ V h_j)
        """
        g = g.local_var()

        g.ndata["e_src"] = self.src_gate(node_feats)
        g.ndata["e_dst"] = self.dst_gate(node_feats)
        g.apply_edges(fn.u_add_v("e_src", "e_dst", "e_nodes"))
        m = g.edata.pop("e_nodes") + self.edge_gate(edge_feats)

        g.edata["sigma"] = torch.sigmoid(m)
        g.ndata["Bh"] = self.dst_update(node_feats)
        g.update_all(
            fn.u_mul_e("Bh", "sigma", "m"), fn.sum("m", "sum_sigma_h")
        )
        g.update_all(fn.copy_e("sigma", "m"), fn.sum("m", "sum_sigma"))
        g.ndata["h"] = g.ndata["sum_sigma_h"] / (g.ndata["sum_sigma"] + 1e-6)
        x = self.src_update(node_feats) + g.ndata.pop("h")

        return x, m


class Conv(nn.Module):
    def __init__(self, in_feat, out_feat, residual=True):
        super().__init__()
        self.residual = residual
        self.gnn = EGGConv(in_feat, out_feat)
        self.bn_nodes = nn.BatchNorm1d(out_feat)
        self.bn_edges = nn.BatchNorm1d(out_feat)

    def forward(self, g, x_in, y_in):
        x, y = self.gnn(g, x_in, y_in)
        x, y = F.silu(self.bn_nodes(x)), F.silu(self.bn_edges(y))

        if self.residual:
            x = x + x_in
            y = y + y_in

        return x, y


class EquivBlock(nn.Module):
    def __init__(self, args, residual=True):
        super().__init__()
        self.residual = residual
        self.W_h = nn.Linear(args.hidden_features, 1)

    def forward(
        self,
        g: dgl.DGLGraph,
        v: torch.Tensor,
        x: torch.Tensor,
    ):
        g = g.local_var()

        g.ndata['x_i'] = self.W_h(x)/256
        g.ndata['x_j'] = -self.W_h(x)/256
        g.apply_edges(fn.u_add_v("x_i", "x_j", "x_nodes"))
        phi = g.edata.pop("x_nodes")
        g.edata['v'] = g.edata['u'] * phi

        if self.residual:
            g.edata['v'] = (v + g.edata['v']) / 2

        return g.edata['v']


class EDiEGGCConv(nn.Module):
    """Line graph update."""

    def __init__(self, args, line_graph: bool, residual: bool = True):
        super().__init__()
        self.line_graph = line_graph
        if self.line_graph:
            self.edge_update = Conv(args.hidden_features, args.hidden_features, residual)

        self.node_update = Conv(args.hidden_features, args.hidden_features, residual)

        self.equiv_block = EquivBlock(args, residual)

    def forward(self, g, lg, v_in, x_in, y_in, z_in):
        g = g.local_var()

        x, y = self.node_update(g, x_in, y_in)
        v = self.equiv_block(g, v_in, x)
        z = z_in
        if self.line_graph:
            lg = lg.local_var()
            # Edge-gated graph convolution update on crystal graph
            y, z = self.edge_update(lg, y, z)

        return v, x, y, z


class MLPLayer(nn.Module):
    """Multilayer perceptron layer helper."""
    def __init__(self, in_features: int, out_features: int):
        """Linear, Batchnorm, SiLU layer."""
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
        )

    def forward(self, x):
        """Linear, Batchnorm, silu layer."""
        return self.layer(x)


class Encoder(nn.Module):
    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()

        self.hidden_features = args.hidden_features
        self.atom_embedding = MLPLayer(args.atom_input_features, args.hidden_features)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=args.cutoff, bins=args.edge_input_features),
            MLPLayer(args.edge_input_features, args.embedding_features),
            MLPLayer(args.embedding_features, args.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=args.triplet_input_features),
            MLPLayer(args.triplet_input_features, args.embedding_features),
            MLPLayer(args.embedding_features, args.hidden_features),
        )
        self.module_layers = nn.ModuleList([EDiEGGCConv(args, True) for idx in range(args.alignn_layers)]
                                           + [EDiEGGCConv(args, False) for idx in range(args.gcn_layers-2)])

    def forward(self, g, lg):
        g = g.local_var()
        lg = lg.local_var()

        # initial node features: atom feature network...
        x = g.ndata.pop("atom_features")
        x = self.atom_embedding(x)

        # initial bond features
        y = self.edge_embedding(g.edata['d'])

        # angle features (fixed)
        z = self.angle_embedding(lg.edata.pop("h"))

        # initial vector features
        v = torch.zeros_like(g.edata['r'])#.view(-1, 3, 1).expand(-1, -1, self.hidden_features))

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            v, x, y, z = module(g, lg, v, x, y, z)

        return v, x, y, z


class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.target = args.target

        self.module_layers = nn.ModuleList([EDiEGGCConv(args, False) for idx in range(2)])
        self.readout = nn.Linear(args.hidden_features, args.output_features)
        self.pooling = AvgPooling()  # SumPooling()
        
        self.q = nn.Sequential(
            MLPLayer(args.hidden_features, args.hidden_features),
            nn.Linear(args.hidden_features, 1),
        )

    def forward(self, g, lg, v, x, y, z):
        g = g.local_var()

        # gated GCN updates: update node, edge features
        for module in self.module_layers:
            v, x, y, z = module(g, lg, v, x, y, z)

        # norm-activation-pool-classify
        if self.target == 'mu':
            g.edata['v'] = v
            g.update_all(fn.copy_e("v", "v"), fn.sum("v", "out"))

            # mu = torch.squeeze(self.readout(v)) / self.hidden_features
            q = self.q(x) #/ self.hidden_features
            h = self.pooling(g, g.ndata['out'] + q * g.ndata['pos'])

            out = torch.norm(h, dim=1)
        else:
            out = self.pooling(g, x)
            out = self.readout(out)

        return torch.squeeze(out)


class EDiEGGC(nn.Module):

    def __init__(self, args):
        """Initialize class with number of input features, conv layers."""
        super().__init__()
        self.target = args.target

        self.encoder = Encoder(args)
        self.decoder = Decoder(args)

    def forward(self, g, lg):
        g = g.local_var()
        lg = lg.local_var()

        v, x, y, z = self.encoder(g, lg)
        out = self.decoder(g, lg, v, x, y, z)

        return out, v, x, y, z

