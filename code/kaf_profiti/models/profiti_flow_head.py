import math
from typing import Dict, Tuple

import torch
from torch import Tensor, nn


class IdentityShiesh(nn.Module):
    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        return z, torch.zeros_like(z)


class TriangularAttention(nn.Module):
    def __init__(self, hidden_dim: int, marginal_training: bool, device: torch.device):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.marginal_training = marginal_training
        self.device = device
        self.scale = hidden_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.05)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.05)
        nn.init.zeros_(self.k_proj.bias)

    def forward(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        if self.marginal_training:
            return self._diagonal(hidden_states, mask)
        return self._full(hidden_states, mask)

    def _full(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        batch_size, query_count, _ = hidden_states.shape
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        scores = torch.bmm(query, key.transpose(-2, -1)) * self.scale
        diagonal = torch.nn.functional.softplus(scores.diagonal(dim1=-2, dim2=-1)) + 1e-3
        off_diag = 0.01 * torch.tanh(torch.tril(scores, diagonal=-1))
        matrix = off_diag + torch.diag_embed(diagonal)
        attention_mask = mask.unsqueeze(-1) * mask.unsqueeze(-2)
        matrix = matrix * attention_mask
        matrix = matrix + torch.diag_embed(1.0 - mask)
        return matrix

    def _diagonal(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        query = self.q_proj(hidden_states)
        key = self.k_proj(hidden_states)
        scores = (query * key).sum(dim=-1) * self.scale
        diagonal = (torch.nn.functional.softplus(scores) + 1e-3) * mask + (1.0 - mask)
        return torch.diag_embed(diagonal)

    @staticmethod
    def log_determinant(attention_matrix: Tensor, mask: Tensor) -> Tensor:
        diagonal = torch.diagonal(attention_matrix, dim1=-2, dim2=-1)
        masked_diagonal = diagonal * mask + (1.0 - mask)
        return torch.log(masked_diagonal.clamp_min(1e-12)).sum(dim=-1)


def _dense_layers(hidden_dim: int) -> nn.Sequential:
    layers = nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )
    nn.init.zeros_(layers[-1].weight)
    nn.init.zeros_(layers[-1].bias)
    return layers


class FlowLayer(nn.Module):
    def __init__(self, hidden_dim: int, marginal_training: bool, device: torch.device):
        super().__init__()
        self.device = device
        self.attention = TriangularAttention(hidden_dim, marginal_training, device)
        self.scale_net = _dense_layers(hidden_dim)
        self.shift_net = _dense_layers(hidden_dim)
        self.tanh = nn.Tanh()
        self.shiesh = IdentityShiesh()
        self.shiesh_inv = IdentityShiesh()

    def forward(self, z: Tensor, hidden_states: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(z.shape[0], device=z.device)
        attention_matrix = self.attention(hidden_states, mask)
        z = torch.bmm(attention_matrix, z.unsqueeze(-1)).squeeze(-1) * mask
        ldj = ldj + self.attention.log_determinant(attention_matrix, mask)

        scale = self.tanh(self.scale_net(hidden_states)).squeeze(-1)
        shift = self.shift_net(hidden_states).squeeze(-1)
        z = (z * torch.exp(scale) + shift) * mask
        ldj = ldj + (scale * mask).sum(dim=-1)
        z, act_ldj = self.shiesh(z)
        ldj = ldj + (act_ldj * mask).sum(dim=-1)
        return z, ldj

    def inverse(self, z: Tensor, hidden_states: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        ldj = torch.zeros(z.shape[0], device=z.device)
        z, act_ldj = self.shiesh_inv(z)
        ldj = ldj + (act_ldj * mask).sum(dim=-1)

        scale = self.tanh(self.scale_net(hidden_states)).squeeze(-1)
        shift = self.shift_net(hidden_states).squeeze(-1)
        z = ((z - shift) / torch.exp(scale)) * mask
        ldj = ldj - (scale * mask).sum(dim=-1)

        attention_matrix = self.attention(hidden_states, mask)
        z = torch.linalg.solve_triangular(
            attention_matrix, z.unsqueeze(-1), upper=False
        ).squeeze(-1)
        z = z * mask
        ldj = ldj - self.attention.log_determinant(attention_matrix, mask)
        return z, ldj


class NormalizingFlow(nn.Module):
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        marginal_training: bool,
        device: torch.device,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [FlowLayer(hidden_dim, marginal_training, device) for _ in range(num_layers)]
        )
        self.el0 = nn.Linear(hidden_dim, 1)

    def forward(self, y: Tensor, hidden_states: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        z = y * mask
        ldj = torch.zeros(y.shape[0], device=y.device)
        offset = self.el0(hidden_states).squeeze(-1)
        z = (z - offset) * mask
        for layer in self.layers:
            z, layer_ldj = layer.forward(z, hidden_states, mask)
            ldj = ldj + layer_ldj
        return z, ldj

    def inverse(self, z: Tensor, hidden_states: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
        y = z * mask
        ldj = torch.zeros(z.shape[0], device=z.device)
        for layer in reversed(self.layers):
            y, layer_ldj = layer.inverse(y, hidden_states, mask)
            ldj = ldj + layer_ldj
        offset = self.el0(hidden_states).squeeze(-1)
        y = (y + offset) * mask
        return y, ldj


class ProFITiFlowHead(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        flow_layers: int,
        marginal_training: bool,
        device: torch.device,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.flow = NormalizingFlow(flow_layers, hidden_dim, marginal_training, self.device)

    def nll(self, y_flat: Tensor, hidden_states: Tensor, mask: Tensor) -> Tensor:
        z, ldj = self.flow.forward(y_flat, hidden_states, mask)
        gaussian_nll = 0.5 * (z.pow(2) + math.log(2.0 * math.pi)) * mask
        joint_nll = gaussian_nll.sum(dim=-1) - ldj
        return joint_nll / mask.sum(dim=-1).clamp_min(1.0)

    def sample(self, hidden_states: Tensor, mask: Tensor, nsamples: int = 100) -> Tensor:
        batch_size, query_count, _ = hidden_states.shape
        z = torch.randn(batch_size, nsamples, query_count, device=hidden_states.device)
        z_flat = z.reshape(batch_size * nsamples, query_count)
        hidden_flat = (
            hidden_states.unsqueeze(1)
            .expand(batch_size, nsamples, query_count, hidden_states.shape[-1])
            .reshape(batch_size * nsamples, query_count, hidden_states.shape[-1])
        )
        mask_flat = (
            mask.unsqueeze(1).expand(batch_size, nsamples, query_count).reshape(batch_size * nsamples, query_count)
        )
        y_flat, _ = self.flow.inverse(z_flat, hidden_flat, mask_flat)
        return y_flat.reshape(batch_size, nsamples, query_count)

    def mean(self, hidden_states: Tensor, mask: Tensor, nsamples: int = 100) -> Tensor:
        return self.sample(hidden_states, mask, nsamples=nsamples).mean(dim=1)

    def base_mean(self, hidden_states: Tensor, mask: Tensor) -> Tensor:
        return self.flow.el0(hidden_states).squeeze(-1) * mask

    @staticmethod
    def masked_mse(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
        return (((yhat - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def masked_mae(y: Tensor, yhat: Tensor, mask: Tensor) -> Tensor:
        return ((yhat - y).abs() * mask).sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def crps(y: Tensor, samples: Tensor, mask: Tensor) -> Tensor:
        term1 = (samples - y.unsqueeze(1)).abs().mean(dim=1)
        pairwise = (samples.unsqueeze(2) - samples.unsqueeze(1)).abs().mean(dim=(1, 2))
        crps = (term1 - 0.5 * pairwise) * mask
        return crps.sum() / mask.sum().clamp_min(1.0)

    @staticmethod
    def energy_score(y: Tensor, samples: Tensor, mask: Tensor, beta: float = 1.0) -> Tensor:
        valid = mask.sum(dim=1) > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=y.device)
        y_valid = y[valid] * mask[valid]
        samples_valid = samples[valid] * mask[valid, None, :]
        first = torch.cdist(samples_valid, y_valid[:, None, :], p=2).pow(beta).mean(dim=1).squeeze(-1)
        pairwise = torch.cdist(samples_valid, samples_valid, p=2).pow(beta)
        second = pairwise.mean(dim=(1, 2)) * 0.5
        return (first - second).mean()
