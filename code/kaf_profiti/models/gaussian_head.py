import math

import torch
from torch import Tensor, nn


class GaussianHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mean = nn.Linear(hidden_dim, 1)
        self.log_var = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.mean.weight)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -1.0)

    def distribution_params(self, hidden_states: Tensor, mask: Tensor):
        mean = self.mean(hidden_states).squeeze(-1) * mask
        log_var = self.log_var(hidden_states).squeeze(-1).clamp(-6.0, 6.0)
        return mean, log_var

    def nll(self, y: Tensor, hidden_states: Tensor, mask: Tensor) -> Tensor:
        mean, log_var = self.distribution_params(hidden_states, mask)
        var = torch.exp(log_var)
        nll = 0.5 * (((y - mean).pow(2) / var) + log_var + math.log(2.0 * math.pi))
        return (nll * mask).sum(dim=-1) / mask.sum(dim=-1).clamp_min(1.0)

    def diagnostics(self, y: Tensor, hidden_states: Tensor, mask: Tensor) -> dict:
        mean, log_var = self.distribution_params(hidden_states, mask)
        nll = self.nll(y, hidden_states, mask)
        denom = mask.sum().detach().cpu().clamp_min(1.0)
        return {
            "y_flat_abs_mean": float((y.abs() * mask).sum().detach().cpu() / denom),
            "hidden_abs_mean": float(hidden_states.abs().mean().detach().cpu()),
            "nll_isfinite": bool(torch.isfinite(nll).all().detach().cpu()),
            "gaussian_log_var_mean": float((log_var * mask).sum().detach().cpu() / denom),
            "gaussian_mean_abs_mean": float((mean.abs() * mask).sum().detach().cpu() / denom),
        }

    def sample(self, hidden_states: Tensor, mask: Tensor, nsamples: int = 100) -> Tensor:
        mean, log_var = self.distribution_params(hidden_states, mask)
        std = torch.exp(0.5 * log_var)
        eps = torch.randn(mean.shape[0], nsamples, mean.shape[1], device=mean.device)
        return (mean[:, None, :] + eps * std[:, None, :]) * mask[:, None, :]

    @staticmethod
    def masked_mse(y: Tensor, mean: Tensor, mask: Tensor) -> Tensor:
        return (((mean - y) ** 2) * mask).sum() / mask.sum().clamp_min(1.0)
