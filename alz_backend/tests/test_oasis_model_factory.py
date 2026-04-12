"""Smoke tests for the OASIS-1 MONAI model factory."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.models.factory import (
    DenseNet3DConfig,
    EmbeddingConfig,
    ModelFactoryError,
    OASISModelConfig,
    build_densenet3d,
    build_model,
    describe_model_config,
    load_oasis_model_config,
)


class _FakeDenseNet121:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


def test_oasis_model_yaml_config_loads_expected_contract() -> None:
    """The default YAML should match the OASIS transform pipeline assumptions."""

    cfg = load_oasis_model_config()
    description = describe_model_config(cfg)

    assert cfg.dataset == "oasis1"
    assert cfg.architecture == "densenet121_3d"
    assert cfg.densenet.spatial_dims == 3
    assert cfg.densenet.in_channels == 1
    assert cfg.densenet.out_channels == 2
    assert cfg.class_names == ("nondemented", "demented")
    assert cfg.expected_input_shape == (1, 1, 128, 128, 128)
    assert description["input_shape_note"] == "Batched OASIS tensors should be shaped (B, C, D, H, W)."


def test_build_model_dispatches_to_densenet3d(monkeypatch: pytest.MonkeyPatch) -> None:
    """Factory dispatch should build the configured MONAI DenseNet3D baseline."""

    monkeypatch.setattr(
        "src.models.factory._load_monai_network_symbols",
        lambda: {"DenseNet121": _FakeDenseNet121},
    )
    cfg = OASISModelConfig(
        expected_input_shape=(1, 1, 64, 64, 64),
        densenet=DenseNet3DConfig(in_channels=1, out_channels=2, dropout_prob=0.25),
    )

    model = build_model(cfg)

    assert isinstance(model, _FakeDenseNet121)
    assert model.kwargs["spatial_dims"] == 3
    assert model.kwargs["in_channels"] == 1
    assert model.kwargs["out_channels"] == 2
    assert model.kwargs["dropout_prob"] == 0.25


def test_build_model_rejects_unsupported_architecture() -> None:
    """Unsupported future architectures should fail clearly until implemented."""

    cfg = OASISModelConfig(architecture="resnet3d")
    with pytest.raises(ModelFactoryError):
        build_model(cfg)


def test_oasis_model_config_loader_rejects_channel_mismatch(tmp_path: Path) -> None:
    """Config validation should catch input-channel mismatches before training."""

    config_path = tmp_path / "oasis_model.yaml"
    config_path.write_text(
        """
architecture: densenet121_3d
expected_input_shape: [1, 2, 64, 64, 64]
densenet:
  in_channels: 1
  out_channels: 2
""",
        encoding="utf-8",
    )

    cfg = load_oasis_model_config(config_path)
    with pytest.raises(ModelFactoryError):
        describe_model_config(cfg)


def test_densenet3d_dummy_tensor_smoke() -> None:
    """The real MONAI DenseNet3D baseline should accept batched 3D MRI tensors."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")

    cfg = DenseNet3DConfig(in_channels=1, out_channels=2, dropout_prob=0.0)
    model = build_densenet3d(cfg)
    model.eval()

    dummy_input = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        logits = model(dummy_input)

    assert tuple(logits.shape) == (1, 2)


def test_embedding_wrapper_returns_penultimate_representation() -> None:
    """Embedding mode should return logits plus a penultimate DenseNet representation."""

    torch = pytest.importorskip("torch")
    pytest.importorskip("monai")

    cfg = OASISModelConfig(
        expected_input_shape=(1, 1, 32, 32, 32),
        densenet=DenseNet3DConfig(in_channels=1, out_channels=2, dropout_prob=0.0),
        embeddings=EmbeddingConfig(enabled=True),
    )
    model = build_model(cfg)
    model.eval()

    dummy_input = torch.zeros((1, 1, 32, 32, 32), dtype=torch.float32)
    with torch.no_grad():
        logits, embeddings = model(dummy_input, return_embeddings=True)

    assert tuple(logits.shape) == (1, 2)
    assert embeddings.shape[0] == 1
    assert embeddings.ndim == 2
