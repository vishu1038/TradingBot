"""MLP: numpy forward vs torch-on-mps forward (run under .venv-ml; skipped elsewhere)."""
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.backends.mps.is_available():
    pytest.skip("mps not available", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
import models as M  # noqa: E402

from tb import features_mtf as fm  # noqa: E402
from tb import model_np as mnp  # noqa: E402

F = len(fm.feature_names(fm.DEFAULT_TFS, ("kalman",)))


@pytest.mark.parametrize("scale", [1.0, 2.5])
def test_mlp_numpy_matches_mps(scale):
    torch.manual_seed(0)
    net, cfg = M.build("mlp", F)
    net = net.to("mps").eval()
    with torch.no_grad():
        for name, p in net.named_parameters():
            p.add_(0.05 * torch.randn_like(p))
            if p.dim() > 1:
                p.mul_(scale)
    x = np.random.default_rng(0).standard_normal((41, F)).astype("float32")
    with torch.no_grad():
        want = net(torch.from_numpy(x).to("mps")).to("cpu").numpy().astype("float64")
    got = mnp.forward("mlp", M.export_state(net), cfg, x)
    assert np.abs(want).max() > 1e-3
    diff = np.abs(got - want).max()
    print(f"parity mlp: max abs diff {diff:.3e}")
    assert diff <= 1e-4 * max(1.0, np.abs(want).max())


def test_mlp_aux_head_columns_do_not_change_main_logits():
    torch.manual_seed(1)
    net, cfg = M.build("mlp", F, cfg={"n_out": 7})
    net = net.to("mps").eval()
    x = np.random.default_rng(1).standard_normal((9, F)).astype("float32")
    with torch.no_grad():
        want = net(torch.from_numpy(x).to("mps")).to("cpu").numpy()
    got = mnp.forward("mlp", M.export_state(net), cfg, x)
    assert got.shape[1] == 7 and np.abs(got[:, :2] - want[:, :2]).max() < 1e-4
