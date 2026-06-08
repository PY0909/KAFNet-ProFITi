from dataclasses import dataclass
from typing import Dict, List

from kaf_profiti.models.kaf_gaussian import KAFGaussian, KAFGaussianConfig
from kaf_profiti.models.kaf_profiti import KAFProFITi, KAFProFITiConfig


@dataclass(frozen=True)
class ModelSpec:
    name: str
    display_name: str
    status: str
    category: str


_MODEL_SPECS: Dict[str, ModelSpec] = {
    "tcn_gaussian": ModelSpec("tcn_gaussian", "TCN-Gaussian", "not_implemented", "baseline"),
    "patchtst_gaussian": ModelSpec(
        "patchtst_gaussian", "PatchTST-Gaussian", "not_implemented", "baseline"
    ),
    "gru_d": ModelSpec("gru_d", "GRU-D", "not_implemented", "baseline"),
    "ode_rnn": ModelSpec("ode_rnn", "ODE-RNN", "not_implemented", "baseline"),
    "mtan": ModelSpec("mtan", "mTAN", "not_implemented", "baseline"),
    "tpatchgnn": ModelSpec("tpatchgnn", "tPatchGNN", "not_implemented", "baseline"),
    "grafiti": ModelSpec("grafiti", "GraFITi", "not_implemented", "baseline"),
    "profiti": ModelSpec("profiti", "ProFITi", "not_implemented", "baseline"),
    "kafnet": ModelSpec("kafnet", "KAFNet", "not_implemented", "baseline"),
    "kafnet_gaussian": ModelSpec(
        "kafnet_gaussian", "KAFNet + Gaussian Head", "enabled", "ablation"
    ),
    "kaf_profiti_marginal": ModelSpec(
        "kaf_profiti_marginal",
        "KAFNet + ProFITi Marginal Flow",
        "enabled",
        "ablation",
    ),
    "kaf_profiti_joint_no_context": ModelSpec(
        "kaf_profiti_joint_no_context",
        "KAFNet + ProFITi Joint Flow w/o Context",
        "enabled",
        "ablation",
    ),
    "kaf_profiti_joint": ModelSpec(
        "kaf_profiti_joint", "KAFNet + ProFITi Joint Flow", "enabled", "final"
    ),
}


def list_model_specs() -> List[ModelSpec]:
    return list(_MODEL_SPECS.values())


def get_model_spec(name: str) -> ModelSpec:
    try:
        return _MODEL_SPECS[name]
    except KeyError as exc:
        raise KeyError(f"Unknown model: {name}") from exc


def create_model(
    name: str,
    num_sensors: int,
    context_dim: int,
    device: str,
    hidden_dim: int = 32,
    te_dim: int = 5,
    kernel_count: int = 4,
    n_layers: int = 2,
    n_heads: int = 2,
    flow_layers: int = 2,
    preconv_dim: int = 8,
    lambda_point: float = 0.1,
):
    spec = get_model_spec(name)
    if spec.status != "enabled":
        raise NotImplementedError(f"Model {name} is registered as {spec.status}")
    if name == "kafnet_gaussian":
        config = KAFGaussianConfig(
            num_sensors=num_sensors,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            device=device,
        )
        return KAFGaussian(config)
    if name == "kaf_profiti_marginal":
        config = KAFProFITiConfig(
            num_sensors=num_sensors,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            flow_layers=flow_layers,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            marginal_training=True,
            device=device,
        )
        return KAFProFITi(config)
    if name == "kaf_profiti_joint_no_context":
        config = KAFProFITiConfig(
            num_sensors=num_sensors,
            context_dim=0,
            hidden_dim=hidden_dim,
            te_dim=te_dim,
            kernel_count=kernel_count,
            n_layers=n_layers,
            n_heads=n_heads,
            flow_layers=flow_layers,
            preconv_dim=preconv_dim,
            lambda_point=lambda_point,
            marginal_training=False,
            device=device,
        )
        return KAFProFITi(config)
    if name != "kaf_profiti_joint":
        raise NotImplementedError(f"Model {name} has no implementation")
    config = KAFProFITiConfig(
        num_sensors=num_sensors,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        te_dim=te_dim,
        kernel_count=kernel_count,
        n_layers=n_layers,
        n_heads=n_heads,
        flow_layers=flow_layers,
        preconv_dim=preconv_dim,
        lambda_point=lambda_point,
        marginal_training=False,
        device=device,
    )
    return KAFProFITi(config)
