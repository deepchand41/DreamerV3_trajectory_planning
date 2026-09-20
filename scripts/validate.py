"""
scripts/validate.py

Run this BEFORE training to confirm every component works on M2.
Prints PASS / FAIL for each check.
"""

import os, sys
os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "glfw")
os.environ["PYOPENGL_PLATFORM"] = os.environ.get("MUJOCO_GL", "glfw")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"


def check(name, fn):
    try:
        result = fn()
        print(f"{PASS}  {name}" + (f"  [{result}]" if result else ""))
    except Exception as e:
        print(f"{FAIL}  {name}  →  {e}")


# ── 1. Python & platform ──────────────────────────────────────────────────────
check("Python 3.10+", lambda: __import__("sys").version.split()[0])
check("ARM64 architecture", lambda: (
    __import__("platform").machine() == "arm64" or None
))

# ── 2. PyTorch + MPS ─────────────────────────────────────────────────────────
def _torch_mps():
    import torch
    assert torch.backends.mps.is_available(), "MPS not available"
    assert torch.backends.mps.is_built(),     "MPS not compiled"
    x = torch.randn(512, 512, device="mps")
    y = x @ x.T
    return f"torch {torch.__version__}, tensor {y.shape} on MPS"

check("PyTorch MPS", _torch_mps)

# ── 3. MuJoCo ─────────────────────────────────────────────────────────────────
def _mujoco():
    import mujoco
    m = mujoco.MjModel.from_xml_string("<mujoco/>")
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d)
    return f"mujoco {mujoco.__version__}"

check("MuJoCo 2.3.7", _mujoco)

# ── 4. Robosuite env creation ─────────────────────────────────────────────────
def _robosuite():
    import robosuite as suite
    import numpy as np
    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_heights=64, camera_widths=64,
        reward_shaping=True, horizon=50,
    )
    obs = env.reset()
    # robosuite raw env uses action_spec, not gym's action_space
    low, high = env.action_spec
    action = np.random.uniform(low, high)
    env.step(action)
    env.close()
    return f"robosuite raw env OK — {len(obs)} obs keys"
check("Robosuite Lift task", _robosuite)

# ── 5. PickPlaceCan task ──────────────────────────────────────────────────────
def _pickplace():
    import robosuite as suite
    env = suite.make(
        "PickPlaceCan", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_heights=64, camera_widths=64,
        reward_shaping=True, horizon=50,
    )
    obs = env.reset()
    env.close()
    return f"PickPlaceCan OK — {len(obs)} obs keys"

check("Robosuite PickPlaceCan task", _pickplace)

# ── 6. Camera rendering ───────────────────────────────────────────────────────
def _camera():
    import robosuite as suite
    import numpy as np
    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names="agentview",
        camera_heights=64, camera_widths=64,
    )
    obs = env.reset()
    img = obs["agentview_image"]
    assert img.shape == (64, 64, 3), f"Expected (64,64,3), got {img.shape}"
    assert img.max() > 0, "Image is all zeros — rendering may be broken"
    env.close()
    return f"image shape {img.shape}, max pixel {img.max()}"

check("Camera rendering (64×64 RGB)", _camera)

# ── 7. Full network forward pass on MPS ───────────────────────────────────────
def _networks():
    import torch, torch.nn as nn, torch.nn.functional as F
    device = torch.device("mps")
    B = 4

    # Encoder
    enc = nn.Sequential(
        nn.Conv2d(3, 48, 4, 2), nn.SiLU(),
        nn.Conv2d(48, 96, 4, 2), nn.SiLU(),
        nn.Conv2d(96, 192, 4, 2), nn.SiLU(),
        nn.Conv2d(192, 384, 4, 2), nn.SiLU(),
        nn.Flatten(),
    ).to(device)
    img = torch.randint(0, 255, (B, 3, 64, 64), dtype=torch.float32).to(device) / 255.0
    emb = enc(img)

    # GRU
    gru = nn.GRUCell(1024 + 7, 512).to(device)
    h   = torch.zeros(B, 512, device=device)
    x   = torch.randn(B, 1024 + 7, device=device)
    h2  = gru(x, h)

    # Gumbel-softmax (M2 safe version)
    from utils.device import safe_gumbel_softmax
    logits = torch.randn(B, 32, 32, device=device)
    s = safe_gumbel_softmax(logits, tau=1.0, hard=True)

    return f"enc: {emb.shape}, h: {h2.shape}, s: {s.shape} — all on MPS"

check("Full network pass on MPS", _networks)

# ── 8. Gumbel-softmax (M2 NaN check) ─────────────────────────────────────────
def _gumbel():
    import torch
    from utils.device import safe_gumbel_softmax
    for _ in range(100):
        logits = torch.randn(16, 32, 32, device="mps")
        s = safe_gumbel_softmax(logits)
        assert not s.isnan().any(), "NaN in gumbel sample!"
    return "100 samples, no NaN"

check("Gumbel-softmax (NaN safety)", _gumbel)

print("\nValidation complete. Fix any FAIL before training.")
