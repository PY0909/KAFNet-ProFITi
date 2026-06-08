from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import Tensor, nn

from .gaussian_head import GaussianHead
from .kafnet_encoder import KAFNetEncoder
from .query_condition_adapter import QueryConditionAdapter


@dataclass
class KAFGaussianConfig:
    num_sensors: int
    context_dim: int
    hidden_dim: int = 32
    te_dim: int = 5
    kernel_count: int = 4
    n_layers: int = 2
    n_heads: int = 2
    preconv_dim: int = 8
    lambda_point: float = 0.1
    device: str = "cpu"

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class KAFGaussian(nn.Module):
    def __init__(self, config: KAFGaussianConfig):
        super().__init__()
        self.config = config
        self.encoder = KAFNetEncoder(
            num_sensors=config.num_sensors,
            hidden_dim=config.hidden_dim,
            kernel_count=config.kernel_count,
            time_dim=config.te_dim,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            preconv_dim=config.preconv_dim,
            context_dim=config.context_dim,
        )
        self.adapter = QueryConditionAdapter(
            num_sensors=config.num_sensors,
            hidden_dim=config.hidden_dim,
            time_dim=config.te_dim,
            context_dim=config.context_dim,
        )
        self.head = GaussianHead(config.hidden_dim)
        self._hidden_states = None
        self.to(torch.device(config.device))

    @property
    def hidden_states(self) -> Tensor:
        if self._hidden_states is None:
            raise RuntimeError("Must call distribution first")
        return self._hidden_states

    def distribution(self, batch) -> Tensor:
        z_var = self.encoder(batch.X_obs, batch.T_obs, batch.M_obs, batch.context)
        self._hidden_states = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            batch.context,
        )
        return self._hidden_states

    def loss(self, batch) -> Tensor:
        hidden = self.distribution(batch)
        nll = self.head.nll(batch.y_flat, hidden, batch.mq_flat).mean()
        if self.config.lambda_point <= 0:
            return nll
        mean, _ = self.head.distribution_params(hidden, batch.mq_flat)
        return nll + self.config.lambda_point * self.head.masked_mse(
            batch.y_flat, mean, batch.mq_flat
        )

    def sample(self, batch, nsamples: int = 100) -> Tensor:
        hidden = self.distribution(batch)
        flat = self.head.sample(hidden, batch.mq_flat, nsamples=nsamples)
        batch_size = batch.X_obs.shape[0]
        pred_len = batch.T_q.shape[1]
        return flat.reshape(batch_size, nsamples, pred_len, self.config.num_sensors)
