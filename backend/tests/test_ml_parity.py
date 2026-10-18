"""numpy forward vs torch-on-mps forward (run under .venv-ml; skipped elsewhere)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("mps not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import models as M  # noqa: E402

from tb import features as ft  # noqa: E402
from tb import model_np as mnp  # noqa: E402


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_numpy_matches_mps(arch):
    torch.manual_seed(0)
    net, cfg = M.build(arch, ft.F, ft.WINDOW)
    net = net.to("mps").eval()
    # non-trivial weights (default init gives near-zero heads, which would hide errors)
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "ln" not in name:
                    p.add_(0.05 * torch.randn_like(p))
    rng = np.random.default_rng(0)
    x = rng.standard_normal((37, ft.WINDOW, ft.F)).astype("float32")
    with torch.no_grad():
        want = net(torch.from_numpy(x).to("mps")).to("cpu").numpy().astype("float64")
    W = {k: v for k, v in M.export_state(net).items()}
    got = mnp.forward(arch, W, cfg, x)
    diff = np.abs(got - want).max()
    print(f"parity {arch}: max abs diff {diff:.3e} (logit scale {np.abs(want).max():.2f})")
    assert np.abs(want).max() > 1e-3
    assert diff <= 1e-4


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_numpy_matches_mps_large_weights(arch):
    """Weights scaled up so every nonlinearity is exercised away from zero."""
    torch.manual_seed(1)
    net, cfg = M.build(arch, ft.F, ft.WINDOW)
    net = net.to("mps").eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if p.dim() > 1 and "ln" not in name:
                p.mul_(2.0)
    rng = np.random.default_rng(5)
    x = rng.standard_normal((16, ft.WINDOW, ft.F)).astype("float32")
    with torch.no_grad():
        want = net(torch.from_numpy(x).to("mps")).to("cpu").numpy().astype("float64")
    got = mnp.forward(arch, M.export_state(net), cfg, x)
    scale = max(1.0, np.abs(want).max())
    assert np.abs(got - want).max() <= 1e-4 * scale


@pytest.mark.parametrize("arch", ["cnn", "gru", "transformer"])
def test_numpy_matches_mps_multitask_head(arch):
    """Auxiliary-head nets (n_out = 7): all seven outputs agree, and the scorer uses columns 0/1 only."""
    torch.manual_seed(2)
    net, cfg = M.build(arch, ft.F, ft.WINDOW, {"n_out": 7})
    net = net.to("mps").eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            if "ln" not in name:
                p.add_(0.05 * torch.randn_like(p))
    x = np.random.default_rng(8).standard_normal((20, ft.WINDOW, ft.F)).astype("float32")
    with torch.no_grad():
        want = net(torch.from_numpy(x).to("mps")).to("cpu").numpy().astype("float64")
    got = mnp.forward(arch, M.export_state(net), cfg, x)
    assert got.shape == want.shape == (20, 7)
    assert np.abs(got - want).max() <= 1e-4
