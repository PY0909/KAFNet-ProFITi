import torch

from kaf_profiti.experiments.registry import create_model, get_model_spec
from kaf_profiti.industrial.batch import IndustrialBatch


def make_batch(batch_size=2, history_len=8, pred_len=3, num_sensors=4, context_dim=3):
    generator = torch.Generator().manual_seed(17)
    x_obs = torch.randn(batch_size, history_len, num_sensors, generator=generator)
    t_obs = torch.arange(history_len, dtype=torch.float32).repeat(batch_size, 1)
    m_obs = (torch.rand(batch_size, history_len, num_sensors, generator=generator) > 0.2).float()
    x_obs = x_obs * m_obs
    t_q = torch.arange(history_len, history_len + pred_len, dtype=torch.float32).repeat(batch_size, 1)
    y_q = torch.randn(batch_size, pred_len, num_sensors, generator=generator)
    m_q = torch.ones(batch_size, pred_len, num_sensors)
    context = torch.randn(batch_size, context_dim, generator=generator)
    return IndustrialBatch(
        X_obs=x_obs,
        T_obs=t_obs,
        M_obs=m_obs,
        T_q=t_q,
        Y_q=y_q,
        M_q=m_q,
        context=context,
        y_flat=y_q.reshape(batch_size, pred_len * num_sensors),
        mq_flat=m_q.reshape(batch_size, pred_len * num_sensors),
        query_channel_ids=torch.arange(num_sensors).repeat(pred_len),
        rul=torch.tensor([50.0, 60.0]),
        unit_id=torch.tensor([1, 2]),
    )


def test_kafnet_gaussian_registry_enabled_and_loss_backward():
    spec = get_model_spec("kafnet_gaussian")
    assert spec.status == "enabled"
    model = create_model(
        "kafnet_gaussian",
        num_sensors=4,
        context_dim=3,
        device="cpu",
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=1,
        preconv_dim=4,
        lambda_point=0.1,
    )
    batch = make_batch()
    loss = model.loss(batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters()
        if p.requires_grad
    )


def test_kafnet_gaussian_samples_shape():
    model = create_model(
        "kafnet_gaussian",
        num_sensors=4,
        context_dim=3,
        device="cpu",
        hidden_dim=16,
        te_dim=5,
        kernel_count=3,
        n_layers=1,
        n_heads=2,
        flow_layers=1,
        preconv_dim=4,
        lambda_point=0.1,
    )
    batch = make_batch()
    samples = model.sample(batch, nsamples=5)
    assert samples.shape == (2, 5, 3, 4)
    assert torch.isfinite(samples).all()


def test_real_baseline_registry_enabled_loss_backward_and_samples():
    for name in ["tcn_gaussian", "gru_d"]:
        spec = get_model_spec(name)
        assert spec.status == "enabled"
        model = create_model(
            name,
            num_sensors=4,
            context_dim=3,
            device="cpu",
            hidden_dim=16,
            te_dim=5,
            kernel_count=3,
            n_layers=1,
            n_heads=2,
            flow_layers=1,
            preconv_dim=4,
            lambda_point=0.1,
        )
        batch = make_batch()
        loss = model.loss(batch)
        loss.backward()
        samples = model.sample(batch, nsamples=5)

        assert torch.isfinite(loss)
        assert samples.shape == (2, 5, 3, 4)
        assert torch.isfinite(samples).all()
        assert any(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.parameters()
            if p.requires_grad
        )


def test_marginal_and_no_context_variants_are_enabled():
    for name in ["kaf_profiti_marginal", "kaf_profiti_joint_no_context"]:
        spec = get_model_spec(name)
        assert spec.status == "enabled"
        model = create_model(
            name,
            num_sensors=4,
            context_dim=3,
            device="cpu",
            hidden_dim=16,
            te_dim=5,
            kernel_count=3,
            n_layers=1,
            n_heads=2,
            flow_layers=1,
            preconv_dim=4,
            lambda_point=0.1,
        )
        batch = make_batch()
        loss = model.loss(batch)
        assert torch.isfinite(loss)
