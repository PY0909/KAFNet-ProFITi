from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from kaf_profiti.industrial.batch import IndustrialCollator
from kaf_profiti.industrial.cmapss import CMapssWindowDataset
from kaf_profiti.industrial.batch import IndustrialBatch
from kaf_profiti.models.kaf_profiti import KAFProFITi, KAFProFITiConfig
from kaf_profiti.models.profiti_flow_head import ProFITiFlowHead
from kaf_profiti.models.query_condition_adapter import QueryConditionAdapter


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _make_batch(batch_size=2, history_len=8, pred_len=3, num_sensors=4):
    generator = torch.Generator().manual_seed(11)
    X_obs = torch.randn(batch_size, history_len, num_sensors, generator=generator)
    T_obs = torch.arange(history_len, dtype=torch.float32).repeat(batch_size, 1)
    M_obs = (torch.rand(batch_size, history_len, num_sensors, generator=generator) > 0.2).float()
    X_obs = X_obs * M_obs
    T_q = torch.arange(history_len, history_len + pred_len, dtype=torch.float32).repeat(batch_size, 1)
    Y_q = torch.randn(batch_size, pred_len, num_sensors, generator=generator)
    M_q = torch.ones(batch_size, pred_len, num_sensors)
    context = torch.randn(batch_size, 3, generator=generator)
    query_channel_ids = torch.arange(num_sensors).repeat(pred_len)
    return IndustrialBatch(
        X_obs=X_obs,
        T_obs=T_obs,
        M_obs=M_obs,
        T_q=T_q,
        Y_q=Y_q,
        M_q=M_q,
        context=context,
        y_flat=Y_q.reshape(batch_size, pred_len * num_sensors),
        mq_flat=M_q.reshape(batch_size, pred_len * num_sensors),
        query_channel_ids=query_channel_ids,
        rul=torch.tensor([50.0, 60.0]),
        unit_id=torch.tensor([1, 2]),
    )


def test_query_condition_adapter_returns_time_first_query_states():
    adapter = QueryConditionAdapter(
        num_sensors=4,
        hidden_dim=16,
        time_dim=5,
        context_dim=3,
        max_len=16,
    )
    z_var = torch.randn(2, 4, 16)
    T_q = torch.tensor([[8.0, 9.0, 10.0], [8.0, 9.0, 10.0]])
    channel_ids = torch.arange(4).repeat(3)
    context = torch.randn(2, 3)

    h_query = adapter(z_var, T_q, channel_ids, context)

    assert h_query.shape == (2, 12, 16)
    assert torch.equal(adapter.build_time_first_channel_ids(3, device=z_var.device), channel_ids)


def test_profiti_flow_head_computes_finite_nll_and_samples():
    head = ProFITiFlowHead(
        hidden_dim=16,
        flow_layers=2,
        marginal_training=False,
        device=torch.device("cpu"),
    )
    y = torch.randn(2, 12)
    hidden_states = torch.randn(2, 12, 16)
    mask = torch.ones(2, 12)

    nll = head.nll(y, hidden_states, mask)
    samples = head.sample(hidden_states, mask, nsamples=5)

    assert nll.shape == (2,)
    assert torch.isfinite(nll).all()
    assert samples.shape == (2, 5, 12)
    assert torch.isfinite(samples).all()


def test_kaf_profiti_loss_backward_smoke():
    batch = _make_batch()
    config = KAFProFITiConfig(
        num_sensors=4,
        context_dim=3,
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=2,
        preconv_dim=4,
        lambda_point=0.1,
        device="cpu",
    )
    model = KAFProFITi(config)

    loss = model.loss(batch, nsamples_for_point=3)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert torch.isfinite(loss)
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_kaf_profiti_supports_batch_size_different_from_attention_heads():
    batch = _make_batch(batch_size=3)
    config = KAFProFITiConfig(
        num_sensors=4,
        context_dim=3,
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=1,
        preconv_dim=4,
        lambda_point=0.0,
        device="cpu",
    )
    model = KAFProFITi(config)

    loss = model.loss(batch)

    assert torch.isfinite(loss)


def test_default_point_loss_is_finite_on_training_like_batch():
    torch.manual_seed(42)
    batch = _make_batch(batch_size=8, history_len=30, pred_len=5, num_sensors=21)
    config = KAFProFITiConfig(
        num_sensors=21,
        context_dim=3,
        hidden_dim=32,
        te_dim=5,
        kernel_count=4,
        n_layers=2,
        n_heads=2,
        flow_layers=2,
        preconv_dim=8,
        lambda_point=0.1,
        device="cpu",
    )
    model = KAFProFITi(config)

    loss = model.loss(batch, nsamples_for_point=1)

    assert torch.isfinite(loss)


def test_default_point_loss_is_finite_on_real_cmapss_shuffle_batch():
    torch.manual_seed(42)
    data_dir = PROJECT_ROOT / "dataset" / "CMAPSSData"
    if not (data_dir / "train_FD001.txt").exists():
        pytest.skip(f"C-MAPSS FD001 data not found: {data_dir}")
    dataset = CMapssWindowDataset(
        data_dir,
        subset="FD001",
        split="train",
        history_len=30,
        pred_len=5,
        stride=1,
        async_mode="mixed",
        seed=42,
    )
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=IndustrialCollator(),
    )
    config = KAFProFITiConfig(
        num_sensors=21,
        context_dim=3,
        hidden_dim=32,
        te_dim=5,
        kernel_count=4,
        n_layers=2,
        n_heads=2,
        flow_layers=2,
        preconv_dim=8,
        lambda_point=0.1,
        device="cpu",
    )
    model = KAFProFITi(config)
    batch = next(iter(loader))

    loss = model.loss(batch, nsamples_for_point=1)

    assert torch.isfinite(loss)
