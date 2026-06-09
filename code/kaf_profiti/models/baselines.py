from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn

from .gaussian_head import GaussianHead
from .query_condition_adapter import QueryConditionAdapter


@dataclass
class TCNGaussianConfig:
    num_sensors: int
    context_dim: int
    hidden_dim: int = 32
    te_dim: int = 5
    n_layers: int = 2
    lambda_point: float = 0.1
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)


@dataclass
class GRUDGaussianConfig:
    num_sensors: int
    context_dim: int
    hidden_dim: int = 32
    te_dim: int = 5
    n_layers: int = 1
    lambda_point: float = 0.1
    device: str = "cpu"

    def to_dict(self):
        return asdict(self)


class TemporalConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=dilation, dilation=dilation),
            nn.ReLU(True),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.net(x.transpose(1, 2)).transpose(1, 2)
        return self.norm(out + residual)


class _GaussianForecastMixin:
    head: GaussianHead
    config: object

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


class TCNGaussian(_GaussianForecastMixin, nn.Module):
    def __init__(self, config: TCNGaussianConfig):
        super().__init__()
        self.config = config
        input_dim = 2 * config.num_sensors + 1
        self.input_proj = nn.Linear(input_dim, config.hidden_dim)
        self.blocks = nn.ModuleList(
            [TemporalConvBlock(config.hidden_dim, dilation=2**idx) for idx in range(config.n_layers)]
        )
        self.sensor_embedding = nn.Embedding(config.num_sensors, config.hidden_dim)
        self.context_proj = (
            nn.Linear(config.context_dim, config.hidden_dim) if config.context_dim > 0 else None
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

    def _encode_history(self, batch) -> Tensor:
        t0 = batch.T_obs[:, :1]
        span = (batch.T_obs[:, -1:] - t0).clamp_min(1.0)
        time_feature = ((batch.T_obs - t0) / span).unsqueeze(-1)
        features = torch.cat([batch.X_obs, batch.M_obs, time_feature], dim=-1)
        hidden = self.input_proj(features)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden[:, -1]

    def distribution(self, batch) -> Tensor:
        global_state = self._encode_history(batch)
        sensor_ids = torch.arange(self.config.num_sensors, device=global_state.device)
        z_var = global_state[:, None, :] + self.sensor_embedding(sensor_ids).unsqueeze(0)
        if self.context_proj is not None:
            z_var = z_var + self.context_proj(batch.context).unsqueeze(1)
        self._hidden_states = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            batch.context,
        )
        return self._hidden_states


class GRUDGaussian(_GaussianForecastMixin, nn.Module):
    def __init__(self, config: GRUDGaussianConfig):
        super().__init__()
        self.config = config
        self.feature_mean = nn.Parameter(torch.zeros(config.num_sensors))
        self.decay_x_weight = nn.Parameter(torch.ones(config.num_sensors))
        self.decay_x_bias = nn.Parameter(torch.zeros(config.num_sensors))
        self.decay_h = nn.Linear(config.num_sensors, config.hidden_dim)
        self.gru_cell = nn.GRUCell(3 * config.num_sensors, config.hidden_dim)
        self.post_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(config.hidden_dim),
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.ReLU(True),
                )
                for _ in range(max(0, config.n_layers - 1))
            ]
        )
        self.sensor_embedding = nn.Embedding(config.num_sensors, config.hidden_dim)
        self.context_proj = (
            nn.Linear(config.context_dim, config.hidden_dim) if config.context_dim > 0 else None
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

    def _deltas(self, t_obs: Tensor, mask: Tensor) -> Tensor:
        batch_size, history_len, num_sensors = mask.shape
        deltas = torch.zeros(batch_size, history_len, num_sensors, device=mask.device)
        for idx in range(1, history_len):
            gap = (t_obs[:, idx] - t_obs[:, idx - 1]).clamp_min(0.0).unsqueeze(-1)
            deltas[:, idx] = gap + (1.0 - mask[:, idx - 1]) * deltas[:, idx - 1]
        return deltas

    def _encode_history(self, batch) -> Tensor:
        batch_size, history_len, _ = batch.X_obs.shape
        deltas = self._deltas(batch.T_obs, batch.M_obs)
        last_obs = self.feature_mean.unsqueeze(0).expand(batch_size, -1)
        hidden = torch.zeros(batch_size, self.config.hidden_dim, device=batch.X_obs.device)
        decay_x_weight = torch.nn.functional.softplus(self.decay_x_weight)
        for idx in range(history_len):
            mask_t = batch.M_obs[:, idx]
            raw_x_t = batch.X_obs[:, idx]
            delta_t = deltas[:, idx]
            gamma_x = torch.exp(-torch.relu(delta_t * decay_x_weight + self.decay_x_bias))
            imputed = mask_t * raw_x_t + (1.0 - mask_t) * (
                gamma_x * last_obs + (1.0 - gamma_x) * self.feature_mean.unsqueeze(0)
            )
            last_obs = torch.where(mask_t.bool(), raw_x_t, last_obs)
            gamma_h = torch.exp(-torch.relu(self.decay_h(delta_t)))
            hidden = hidden * gamma_h
            hidden = self.gru_cell(torch.cat([imputed, mask_t, delta_t], dim=-1), hidden)
        for layer in self.post_layers:
            hidden = hidden + layer(hidden)
        return hidden

    def distribution(self, batch) -> Tensor:
        global_state = self._encode_history(batch)
        sensor_ids = torch.arange(self.config.num_sensors, device=global_state.device)
        z_var = global_state[:, None, :] + self.sensor_embedding(sensor_ids).unsqueeze(0)
        if self.context_proj is not None:
            z_var = z_var + self.context_proj(batch.context).unsqueeze(1)
        self._hidden_states = self.adapter(
            z_var,
            batch.T_q,
            batch.query_channel_ids,
            batch.context,
        )
        return self._hidden_states
