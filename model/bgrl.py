import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from model.utils import BentIdentity
from model.edieggc import Encoder, Decoder


def get_module_device(module):
    return next(module.parameters()).device


def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val


def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


# exponential moving average

class EMA():
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new


def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class BGRL(nn.Module):
    def __init__(self, args,
                 predictor_dim=256,
                 moving_average_decay=0.99,
                 ):
        super().__init__()
        self.online_encoder = Encoder(args)
        self.online_decoder = Decoder(args)
        self.target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(self.target_encoder, False)
        self.target_ema_updater = EMA(moving_average_decay, args.epochs)
        rep_dim = args.hidden_features
        self.online_predictor = nn.Sequential(
            nn.Linear(rep_dim, predictor_dim),
            BentIdentity(),
            nn.Linear(predictor_dim, rep_dim)
        )
        self.online_predictor.apply(init_weights)

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.target_encoder is not None, 'Target encoder has not been created yet'
        update_moving_average(self.target_ema_updater, self.target_encoder, self.online_encoder)

    def forward(self, g1, lg1, g2, lg2):
        online_proj_1_v, online_proj_1_x, online_proj_1_y, online_proj_1_z = self.online_encoder(g1, lg1)
        online_proj_2_v, online_proj_2_x, online_proj_2_y, online_proj_2_z = self.online_encoder(g2, lg2)

        online_pred_1 = self.online_predictor(online_proj_1_y)
        online_pred_2 = self.online_predictor(online_proj_2_y)

        with torch.no_grad():
            _, _, target_proj_1, _ = self.target_encoder(g1, lg1)
            _, _, target_proj_2, _ = self.target_encoder(g2, lg2)

        loss_1 = loss_fn(online_pred_1, target_proj_2.detach())
        loss_2 = loss_fn(online_pred_2, target_proj_1.detach())

        loss = loss_1 + loss_2

        pred_1 = self.online_decoder(g1, lg1, online_proj_1_v, online_proj_1_x, online_proj_1_y, online_proj_1_z)
        pred_2 = self.online_decoder(g2, lg2, online_proj_2_v, online_proj_2_x, online_proj_2_y, online_proj_2_z)
        return pred_1, pred_2, loss.mean(), online_proj_1_v, online_proj_1_x, online_proj_1_y, online_proj_1_z

