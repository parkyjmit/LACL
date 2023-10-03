'''
QMugs dataset: https://www.research-collection.ethz.ch/handle/20.500.11850/482129
QM9 dataset: https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904
'''

from torch.utils.data import DataLoader
from jarvis.core.specie import get_node_attributes
from jarvis.core.graphs import compute_bond_cosines
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import dgl
import random
import logging
from tqdm import tqdm
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from joblib import Parallel, delayed


table = {1: 'H', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 14: 'P', 15: 'Si', 16: 'S', 17: 'Cl', 35: 'Br', 53: 'I'}
torch.manual_seed(123)
np.random.seed(123)


def compute_d_u(edges):
    r = edges.dst['pos'] - edges.src['pos']
    d = torch.norm(r, dim=1)
    u = r/d[:, None]
    return {'r': r, 'd': d, 'u': u}


def pos_z_to_graph(pos, z, cutoff):
    '''
    position, atomic number -> dgl.graph
    :param pos: np.array
    :param z: np.array
    :param cutoff: float
    :return: dgl.graph
    '''
    D_st = np.linalg.norm(pos[None, :] - pos[:, None], axis=-1)
    edges_idx = (sp.csr_matrix(D_st <= cutoff) - sp.eye(z.shape[0], dtype=bool)).nonzero()
    g = dgl.graph(edges_idx, num_nodes=z.shape[0])

    g.ndata['pos'] = torch.tensor(pos, dtype=torch.float32)
    g.apply_edges(compute_d_u)
    g.ndata['atom_numbers'] = torch.tensor(z, dtype=torch.int32)
    # build up atom attribute tensor
    sps_features = []
    for s in g.ndata['atom_numbers']:
        feat = list(get_node_attributes(table[int(s)], atom_features='cgcnn'))
        sps_features.append(feat)
    sps_features = np.array(sps_features)
    node_features = torch.tensor(sps_features).type(
        torch.get_default_dtype()
    )
    g.ndata["atom_features"] = node_features
    return g


def dict_to_graph(mol, cutoff):
    '''
    :param mol: input molecule dictionary
    :param cutoff: cutoff radius
    :return: dgl.graph
    '''
    pos = mol['pos']
    z = mol['z']
    return pos_z_to_graph(pos, z, cutoff)


def dft_mmff_to_graph(smiles, g, cutoff):
    # make molecule
    g = g.local_var()
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.Compute2DCoords(mol)

    # conformation optimization MMFF
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    newpos = g.ndata['pos']  # from DFT
    for i in range(num_atoms):
        conf.SetAtomPosition(i, Point3D(newpos[i][0].item(), newpos[i][1].item(), newpos[i][2].item()))
    AllChem.MMFFOptimizeMolecule(mol, confId=0)

    return rdkmol_to_graph(mol, g, cutoff)


def rdkmol_to_graph(mol, g, cutoff):
    # conformation to graph
    g = g.local_var()
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    pos = np.zeros([num_atoms, 3])
    z = []
    for i in range(num_atoms):
        position = conf.GetAtomPosition(i)
        pos[i] = np.array([position.x, position.y, position.z])
        z.append(mol.GetAtomWithIdx(i).GetAtomicNum())
    z = np.array(z)
    g.ndata['pos'] = torch.tensor(pos, dtype=torch.float32)
    g.apply_edges(compute_d_u)
    return g


def prepare_line_dgl(g):
    lg = g.line_graph(shared=True)
    lg.apply_edges(compute_bond_cosines)
    lg.ndata.pop('r')
    lg.ndata.pop('d')
    lg.ndata.pop('u')
    return lg


class MMFFQM9Dataset(torch.utils.data.Dataset):
    '''
    open QM9 pickle file containing dictionary
    dataset which get dgl.graph, dgl.graph from smiles, label
    '''
    def __init__(self, num=133855, cutoff=8.0, target=[], num_workers=7):
        super().__init__()
        self.graphs = []
        self.smiles = []
        self.graphs_line = []
        self.smiles_line = []
        self.labels = []

        with open('data/QM9/qm9_all.pkl', 'rb') as f:
            qm9 = pickle.load(f)
        if num < 100000:
            for i in tqdm(range(num)):
                g = dict_to_graph(qm9[i], cutoff)
                self.graphs.append(g)
                self.smiles.append(dft_mmff_to_graph(qm9[i]['SMILES'], g, cutoff))
                self.labels.append(qm9[i][target])
        else:
            qm9_split = np.array_split(qm9, num_workers)
            def pickle_to_graphs(split):
                graphs = []
                smiles = []
                labels = []
                for d in split:
                    g = dict_to_graph(d, cutoff)
                    graphs.append(g)
                    smiles.append(dft_mmff_to_graph(d['SMILES'], g, cutoff))
                    labels.append(d[target])
                return graphs, smiles, labels

            pickle_list = Parallel(n_jobs=num_workers)(delayed(pickle_to_graphs)(split) for split in qm9_split)
            for g_list, sg_list, l_list in pickle_list:
                self.graphs += g_list
                self.smiles += sg_list
                self.labels += l_list

        logging.info('Building line graphs')
        for i in tqdm(range(len(self.graphs))):
            lg1 = prepare_line_dgl(self.graphs[i])
            self.graphs_line.append(lg1)
            lg2 = prepare_line_dgl(self.smiles[i])
            self.smiles_line.append(lg2)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        g1 = self.graphs[index]
        lg1 = self.graphs_line[index]
        g2 = self.smiles[index]
        lg2 = self.smiles_line[index]
        labels = self.labels[index]
        return g1, lg1, g2, lg2, labels
    
    @staticmethod
    def collate(samples):
        g1, lg1, g2, lg2, labels = map(list, zip(*samples))
        g1 = dgl.batch(g1)
        lg1 = dgl.batch(lg1)
        g2 = dgl.batch(g2)
        lg2 = dgl.batch(lg2)
        return g1, lg1, g2, lg2, torch.tensor(labels)


class CGCFQM9Dataset(torch.utils.data.Dataset):
    '''
    open QM9 pickle file containing dictionary
    dataset which get dgl.graph, dgl.graph from smiles, label
    '''
    def __init__(self, num=133855, cutoff=8.0, target=[], num_workers=7):
        super().__init__()
        self.graphs = []
        self.smiles = []
        self.graphs_line = []
        self.smiles_line = []
        self.labels = []

        with open('data/QM9/qm9_all.pkl', 'rb') as f:
            qm9 = pickle.load(f)
        with open('data/QM9/qm9_all_cgcf.pkl', 'rb') as f:
            qm9_cgcf = pickle.load(f)
        if num < 100000:
            for i in tqdm(range(num)):
                g = dict_to_graph(qm9[i], cutoff)
                self.graphs.append(g)
                self.smiles.append(rdkmol_to_graph(qm9_cgcf[i], g, cutoff))
                self.labels.append(qm9[i][target])
        else:
            qm9_split = np.array_split(qm9, num_workers)
            qm9_cgcf_split = np.array_split(qm9_cgcf, num_workers)
            def pickle_to_graphs(split1, split2):
                graphs = []
                smiles = []
                labels = []
                for d1, d2 in zip(split1, split2):
                    g = dict_to_graph(d1, cutoff)
                    graphs.append(g)
                    smiles.append(rdkmol_to_graph(d2, g, cutoff))
                    labels.append(d1[target])
                return graphs, smiles, labels

            pickle_list = Parallel(n_jobs=num_workers)(delayed(pickle_to_graphs)(split1, split2) for split1, split2 in zip(qm9_split, qm9_cgcf_split))
            for g_list, sg_list, l_list in pickle_list:
                self.graphs += g_list
                self.smiles += sg_list
                self.labels += l_list

        logging.info('Building line graphs')
        for i in tqdm(range(len(self.graphs))):
            lg1 = prepare_line_dgl(self.graphs[i])
            self.graphs_line.append(lg1)
            lg2 = prepare_line_dgl(self.smiles[i])
            self.smiles_line.append(lg2)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        g1 = self.graphs[index]
        lg1 = self.graphs_line[index]
        g2 = self.smiles[index]
        lg2 = self.smiles_line[index]
        labels = self.labels[index]
        return g1, lg1, g2, lg2, labels

    @staticmethod
    def collate(samples):
        g1, lg1, g2, lg2, labels = map(list, zip(*samples))
        g1 = dgl.batch(g1)
        lg1 = dgl.batch(lg1)
        g2 = dgl.batch(g2)
        lg2 = dgl.batch(lg2)
        return g1, lg1, g2, lg2, torch.tensor(labels)


def QM9Dataloader(args):
    '''
    QM9 dataloader.
    :param args: arguments
    :return: graph dataloader
    '''

    num = args.num_train + args.num_valid + args.num_test
    idx = list(range(num))
    random.seed(123)
    random.shuffle(idx)
    train_indices = idx[0:args.num_train]
    valid_indices = idx[args.num_train:args.num_train + args.num_valid]
    test_indices = idx[args.num_train + args.num_valid:args.num_train + args.num_valid + args.num_test]

    logging.info('Prepare train/validation/test data')
    if args.geometry == 'MMFF':
        data = MMFFQM9Dataset(num, cutoff=args.cutoff, target=args.target, num_workers=args.num_workers)
    elif args.geometry == 'CGCF':
        data = CGCFQM9Dataset(num, cutoff=args.cutoff, target=args.target, num_workers=args.num_workers)
    collate_fn = data.collate

    train_data = torch.utils.data.Subset(data, train_indices)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    valid_data = torch.utils.data.Subset(data, valid_indices)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    test_data = torch.utils.data.Subset(data, test_indices)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, (train_indices, valid_indices, test_indices)


class QMugs_Dataset(torch.utils.data.Dataset):
    '''
    ['CHEMBL_ID', 'CONF_ID', 'GFN2:TOTAL_ENERGY', 'GFN2:ATOMIC_ENERGY',
       'GFN2:FORMATION_ENERGY', 'GFN2:TOTAL_ENTHALPY',
       'GFN2:TOTAL_FREE_ENERGY', 'GFN2:DIPOLE', 'GFN2:QUADRUPOLE',
       'GFN2:ROT_CONSTANTS', 'GFN2:ENTHALPY', 'GFN2:HEAT_CAPACITY',
       'GFN2:ENTROPY', 'GFN2:HOMO_ENERGY', 'GFN2:LUMO_ENERGY',
       'GFN2:HOMO_LUMO_GAP', 'GFN2:FERMI_LEVEL', 'GFN2:MULLIKEN_CHARGES',
       'GFN2:COVALENT_COORDINATION_NUMBER',
       'GFN2:DISPERSION_COEFFICIENT_MOLECULAR',
       'GFN2:DISPERSION_COEFFICIENT_ATOMIC', 'GFN2:POLARIZABILITY_MOLECULAR',
       'GFN2:POLARIZABILITY_ATOMIC', 'GFN2:WIBERG_BOND_ORDER',
       'GFN2:TOTAL_WIBERG_BOND_ORDER', 'DFT:TOTAL_ENERGY', 'DFT:ATOMIC_ENERGY',
       'DFT:FORMATION_ENERGY', 'DFT:ESP_AT_NUCLEI', 'DFT:LOWDIN_CHARGES',
       'DFT:MULLIKEN_CHARGES', 'DFT:ROT_CONSTANTS', 'DFT:DIPOLE',
       'DFT:XC_ENERGY', 'DFT:NUCLEAR_REPULSION_ENERGY',
       'DFT:ONE_ELECTRON_ENERGY', 'DFT:TWO_ELECTRON_ENERGY', 'DFT:HOMO_ENERGY',
       'DFT:LUMO_ENERGY', 'DFT:HOMO_LUMO_GAP', 'DFT:MAYER_BOND_ORDER',
       'DFT:WIBERG_LOWDIN_BOND_ORDER', 'DFT:TOTAL_MAYER_BOND_ORDER',
       'DFT:TOTAL_WIBERG_LOWDIN_BOND_ORDER', 'ID', 'SMILES'],
    '''
    def __init__(self, num=204249, relaxation='MMFF', cutoff=8.0, target='DFT:DIPOLE', num_workers=8):
        super().__init__()
        self.cutoff = cutoff
        self.smiles = []
        self.graphs = []
        self.graphs_line = []
        self.smiles_line = []
        self.labels = []

        with open('data/QMugs/QMugs_20_energy.pkl', 'rb') as f:
            QMugs_20_energy = pickle.load(f)
        if relaxation == 'MMFF':
            with open('data/QMugs/QMugs_20_energy_mmff.pkl', 'rb') as f:
                QMugs_20_energy_sg = pickle.load(f)
        elif relaxation == 'CGCF':
            with open('data/QMugs/QMugs_20_energy_cgcf.pkl', 'rb') as f:
                QMugs_20_energy_sg = pickle.load(f)
        
        qmugs_df = pd.concat(QMugs_20_energy[:num])
        QMugs_20_energy_split = np.array_split(qmugs_df['Molecule'], num_workers)
        QMugs_20_energy_sg_split = np.array_split(QMugs_20_energy_sg[:num], num_workers)
        
        def preprocess(split, m_split, cutoff):
            graphs = []
            smiles = []
            for mol, m_mol in zip(split, m_split):
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                g = pos_z_to_graph(pos, z, cutoff)
                graphs.append(g)
                smiles.append(rdkmol_to_graph(m_mol, g, cutoff))
            return graphs, smiles
        preprocessed_list = Parallel(n_jobs=num_workers)(delayed(preprocess)(split, m_split, self.cutoff) 
                                                         for split, m_split in zip(QMugs_20_energy_split, QMugs_20_energy_sg_split))
        for g_list, s_list in preprocessed_list:
            self.graphs += g_list
            self.smiles += s_list

        qmugs_df = pd.concat(QMugs_20_energy[:num])
        if 'DIPOLE' in target:
            self.labels = np.array(qmugs_df[target].apply(lambda x: float(x.split('|')[-1])), dtype=np.float32)
        else:
            self.labels = np.array(qmugs_df[target].apply(float), dtype=np.float32)

        logging.info('Building line graphs')
        for i in tqdm(range(len(self.graphs))):
            lg1 = prepare_line_dgl(self.graphs[i])
            self.graphs_line.append(lg1)
            lg2 = prepare_line_dgl(self.smiles[i])
            self.smiles_line.append(lg2)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        g1 = self.graphs[index]
        lg1 = self.graphs_line[index]
        g2 = self.smiles[index]
        lg2 = self.smiles_line[index]
        labels = self.labels[index]
        return g1, lg1, g2, lg2, labels

    @staticmethod
    def collate(samples):
        g1, lg1, g2, lg2, labels = map(list, zip(*samples))
        g1 = dgl.batch(g1)
        lg1 = dgl.batch(lg1)
        g2 = dgl.batch(g2)
        lg2 = dgl.batch(lg2)
        return g1, lg1, g2, lg2, torch.tensor(labels)


def QMugsDataloader(args):
    '''
    QMugs dataloader.
    :param args: arguments
    :return: graph dataloader
    '''

    num = args.num_train + args.num_valid + args.num_test
    idx = list(range(num))
    random.seed(123)
    random.shuffle(idx)
    train_indices = idx[0:args.num_train]
    valid_indices = idx[args.num_train:args.num_train + args.num_valid]
    test_indices = idx[args.num_train + args.num_valid:args.num_train + args.num_valid + args.num_test]

    logging.info('Prepare train/validation/test data')
    data = QMugs_Dataset(num, cutoff=args.cutoff, relaxation=args.geometry, target=args.target, num_workers=args.num_workers)
    collate_fn = data.collate

    train_data = torch.utils.data.Subset(data, train_indices)
    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    valid_data = torch.utils.data.Subset(data, valid_indices)
    valid_loader = DataLoader(valid_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    test_data = torch.utils.data.Subset(data, test_indices)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             collate_fn=collate_fn)

    return train_loader, valid_loader, test_loader, (train_indices, valid_indices, test_indices)



class QMugs_Test_Dataset(torch.utils.data.Dataset):
    '''
    ['CHEMBL_ID', 'CONF_ID', 'GFN2:TOTAL_ENERGY', 'GFN2:ATOMIC_ENERGY',
       'GFN2:FORMATION_ENERGY', 'GFN2:TOTAL_ENTHALPY',
       'GFN2:TOTAL_FREE_ENERGY', 'GFN2:DIPOLE', 'GFN2:QUADRUPOLE',
       'GFN2:ROT_CONSTANTS', 'GFN2:ENTHALPY', 'GFN2:HEAT_CAPACITY',
       'GFN2:ENTROPY', 'GFN2:HOMO_ENERGY', 'GFN2:LUMO_ENERGY',
       'GFN2:HOMO_LUMO_GAP', 'GFN2:FERMI_LEVEL', 'GFN2:MULLIKEN_CHARGES',
       'GFN2:COVALENT_COORDINATION_NUMBER',
       'GFN2:DISPERSION_COEFFICIENT_MOLECULAR',
       'GFN2:DISPERSION_COEFFICIENT_ATOMIC', 'GFN2:POLARIZABILITY_MOLECULAR',
       'GFN2:POLARIZABILITY_ATOMIC', 'GFN2:WIBERG_BOND_ORDER',
       'GFN2:TOTAL_WIBERG_BOND_ORDER', 'DFT:TOTAL_ENERGY', 'DFT:ATOMIC_ENERGY',
       'DFT:FORMATION_ENERGY', 'DFT:ESP_AT_NUCLEI', 'DFT:LOWDIN_CHARGES',
       'DFT:MULLIKEN_CHARGES', 'DFT:ROT_CONSTANTS', 'DFT:DIPOLE',
       'DFT:XC_ENERGY', 'DFT:NUCLEAR_REPULSION_ENERGY',
       'DFT:ONE_ELECTRON_ENERGY', 'DFT:TWO_ELECTRON_ENERGY', 'DFT:HOMO_ENERGY',
       'DFT:LUMO_ENERGY', 'DFT:HOMO_LUMO_GAP', 'DFT:MAYER_BOND_ORDER',
       'DFT:WIBERG_LOWDIN_BOND_ORDER', 'DFT:TOTAL_MAYER_BOND_ORDER',
       'DFT:TOTAL_WIBERG_LOWDIN_BOND_ORDER', 'ID', 'SMILES'],
    '''
    def __init__(self, typeof='2040', relaxation='MMFF', cutoff=8.0, target='DFT:DIPOLE', num_workers=8):
        super().__init__()
        self.cutoff = cutoff
        self.smiles = []
        self.graphs = []
        self.graphs_line = []
        self.smiles_line = []
        self.labels = []

        with open(f'data/QMugs/QMugs_{typeof}_energy_test.pkl', 'rb') as f:
            QMugs_energy = pickle.load(f)
        with open(f'data/QMugs/QMugs_{typeof}_energy_mmff.pkl', 'rb') as f:
            QMugs_energy_sg = pickle.load(f)
        
        qmugs_df = pd.concat(QMugs_energy)
        QMugs_20_energy_split = np.array_split(qmugs_df['Molecule'], num_workers)
        QMugs_20_energy_sg_split = np.array_split(QMugs_energy_sg, num_workers)
        
        def preprocess(split, m_split, cutoff):
            graphs = []
            smiles = []
            for mol, m_mol in zip(split, m_split):
                conf = mol.GetConformer()
                pos = conf.GetPositions()
                z = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()])
                g = pos_z_to_graph(pos, z, cutoff)
                graphs.append(g)
                smiles.append(rdkmol_to_graph(m_mol, g, cutoff))
            return graphs, smiles
        preprocessed_list = Parallel(n_jobs=num_workers)(delayed(preprocess)(split, m_split, self.cutoff) 
                                                         for split, m_split in zip(QMugs_20_energy_split, QMugs_20_energy_sg_split))
        for g_list, s_list in preprocessed_list:
            self.graphs += g_list
            self.smiles += s_list

        qmugs_df = pd.concat(QMugs_energy)
        if 'DIPOLE' in target:
            self.labels = np.array(qmugs_df[target].apply(lambda x: float(x.split('|')[-1])), dtype=np.float32)
        else:
            self.labels = np.array(qmugs_df[target].apply(float), dtype=np.float32)

        logging.info('Building line graphs')
        for i in tqdm(range(len(self.graphs))):
            lg1 = prepare_line_dgl(self.graphs[i])
            self.graphs_line.append(lg1)
            lg2 = prepare_line_dgl(self.smiles[i])
            self.smiles_line.append(lg2)

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        g1 = self.graphs[index]
        lg1 = self.graphs_line[index]
        g2 = self.smiles[index]
        lg2 = self.smiles_line[index]
        labels = self.labels[index]
        return g1, lg1, g2, lg2, labels

    @staticmethod
    def collate(samples):
        g1, lg1, g2, lg2, labels = map(list, zip(*samples))
        g1 = dgl.batch(g1)
        lg1 = dgl.batch(lg1)
        g2 = dgl.batch(g2)
        lg2 = dgl.batch(lg2)
        return g1, lg1, g2, lg2, torch.tensor(labels)


def QMugsTestDataloader(args):
    '''
    QMugs dataloader.
    :param args: arguments
    :return: graph dataloader
    '''

    logging.info('Prepare train/validation/test data')
    data = QMugs_Test_Dataset(args.typeof, cutoff=args.cutoff, relaxation=args.geometry, target=args.target, num_workers=args.num_workers)
    collate_fn = data.collate

    test_loader = DataLoader(data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    return test_loader