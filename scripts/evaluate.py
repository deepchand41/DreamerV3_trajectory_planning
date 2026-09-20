"""
scripts/evaluate.py

Loads a trained checkpoint and evaluates the policy.
Produces:
  1. Console report  — success rate, mean reward, episode stats
  2. MP4 video       — rendered rollout saved to logs/renders/
  3. Progress plot   — reward curve from training history

Usage:
    # Evaluate latest checkpoint (20 episodes, no video)
    MUJOCO_GL=glfw python scripts/evaluate.py

    # Evaluate with video rendering
    MUJOCO_GL=glfw python scripts/evaluate.py --render

    # Evaluate a specific checkpoint
    MUJOCO_GL=glfw python scripts/evaluate.py --checkpoint checkpoints/step_00100000_20250610_1623.pt

    # Quick check — 5 episodes with video
    MUJOCO_GL=glfw python scripts/evaluate.py --episodes 5 --render
"""

import os, sys, argparse, yaml, time
from pathlib import Path
from datetime import datetime

_gl = os.environ.get("MUJOCO_GL", "glfw")
os.environ["MUJOCO_GL"] = _gl
os.environ["PYOPENGL_PLATFORM"] = _gl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.device import get_device, device_info

# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     default="configs/panda_pickplace_m2.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Path to .pt file. Defaults to checkpoints/latest.pt")
    p.add_argument("--episodes",   type=int,   default=20)
    p.add_argument("--device",     default="cpu")
    p.add_argument("--render",     action="store_true",
                   help="Save an MP4 video of the rollout")
    p.add_argument("--render_episodes", type=int, default=3,
                   help="How many episodes to render into video (default 3)")
    p.add_argument("--deterministic", action="store_true",
                   help="Use mean action (no sampling) — cleaner rollouts for video")
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Reuse network definitions from train.py
# (copy-pasted so evaluate.py is self-contained)
# ─────────────────────────────────────────────────────────────────────────────

def safe(x, name=""):
    if torch.isnan(x).any() or torch.isinf(x).any():
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x

class ImageEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth,       4, 2, 0), nn.BatchNorm2d(depth),    nn.SiLU(),
            nn.Conv2d(depth, depth*2, 4, 2, 0), nn.BatchNorm2d(depth*2), nn.SiLU(),
            nn.Conv2d(depth*2,depth*4,4, 2, 0), nn.BatchNorm2d(depth*4), nn.SiLU(),
            nn.Conv2d(depth*4,depth*8,4, 2, 0), nn.BatchNorm2d(depth*8), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 3, 64, 64)).shape[1]
    def forward(self, x):
        return safe(self.net(x.float() / 255.0 - 0.5))

class VectorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
        )
        self.out_dim = out_dim
    def forward(self, x):
        return safe(self.net(x))

class RSSM(nn.Module):
    def __init__(self, deter, stoch, classes, embed_dim, action_dim):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.lat = stoch * classes
        self.gru_in   = nn.Linear(self.lat + action_dim, deter)
        self.gru_norm = nn.LayerNorm(deter)
        self.gru      = nn.GRUCell(deter, deter)
        H = min(deter, 512)
        self.prior = nn.Sequential(nn.Linear(deter, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, stoch*classes))
        self.post  = nn.Sequential(nn.Linear(deter+embed_dim, H), nn.LayerNorm(H), nn.SiLU(), nn.Linear(H, stoch*classes))
    def init_state(self, B, device):
        return (torch.zeros(B, self.deter, device=device),
                torch.zeros(B, self.stoch, self.classes, device=device))
    def feat(self, h, s):
        return torch.cat([h, s.view(s.shape[0], -1)], -1)
    def obs_step(self, h, s, a, e):
        x  = self.gru_norm(self.gru_in(torch.cat([s.view(s.shape[0],-1).clamp(-5,5), a.clamp(-5,5)], -1)))
        h2 = safe(self.gru(safe(x), h)).clamp(-20, 20)
        pos = safe(self.post(torch.cat([h2, safe(e)], -1)).view(-1, self.stoch, self.classes))
        s2  = F.gumbel_softmax(pos.cpu(), tau=1.0, hard=True, dim=-1).to(h2.device)
        return h2, s2

class Actor(nn.Module):
    def __init__(self, feat_dim, action_dim, layers=3, units=256, min_std=0.1):
        super().__init__()
        self.min_std = min_std
        net = [nn.Linear(feat_dim, units), nn.LayerNorm(units), nn.SiLU()]
        for _ in range(layers-1):
            net += [nn.Linear(units, units), nn.LayerNorm(units), nn.SiLU()]
        self.trunk     = nn.Sequential(*net)
        self.mean_head = nn.Linear(units, action_dim)
        self.std_head  = nn.Linear(units, action_dim)
    def forward(self, z):
        h    = self.trunk(safe(z))
        mean = safe(self.mean_head(h)).clamp(-10, 10)
        std  = safe(F.softplus(self.std_head(h))).clamp(self.min_std, 10.0)
        return torch.distributions.Independent(torch.distributions.Normal(mean, std), 1)
    def act(self, z, deterministic=False):
        dist = self.forward(z)
        return dist.mean if deterministic else dist.sample()

# ─────────────────────────────────────────────────────────────────────────────
# Environment (render-capable)
# ─────────────────────────────────────────────────────────────────────────────

def make_env(cfg, record_video=False):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    env = suite.make(
        env_name=cfg["env"]["task"],
        robots=cfg["env"]["robot"],
        has_renderer=False,
        has_offscreen_renderer=True,   # always on — needed for camera obs + video
        use_camera_obs=True,
        camera_names=cfg["env"]["camera"],
        camera_heights=cfg["env"]["image_size"],   # must match training (64)
        camera_widths=cfg["env"]["image_size"],
        reward_shaping=cfg["env"]["reward_shaping"],
        control_freq=cfg["env"]["control_freq"],
        horizon=cfg["env"]["horizon"],
        ignore_done=False,
        controller_configs=load_controller_config(
            default_controller=cfg["env"]["controller"]
        ),
    )
    return env

def get_obs(env, cfg):
    """Extract dict obs from raw robosuite obs_dict."""
    pass  # returned inline in step/reset below

def process_obs(obs_dict, cfg, img_size=64):
    cam_key = cfg["env"]["camera"] + "_image"
    img = obs_dict[cam_key]
    img = np.ascontiguousarray(np.flipud(img)).transpose(2, 0, 1)
    parts = []
    for k in cfg["env"]["obs_keys"]:
        if k in obs_dict:
            parts.append(np.asarray(obs_dict[k], dtype=np.float32).flatten())
    vec = np.concatenate(parts) if parts else np.zeros(1, np.float32)
    return {"image": img.copy(), "vector": vec}

def grab_frame(env, cfg, height=256, width=256):
    """Grab an RGB frame for video recording."""
    cam = cfg["env"]["camera"]
    frame = env.sim.render(height=height, width=width, camera_name=cam)
    return np.flipud(frame)   # robosuite renders upside-down

# ─────────────────────────────────────────────────────────────────────────────
# Load checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def load_policy(ckpt_path, cfg, device):
    """Load just the actor + encoders + RSSM needed for rollout."""
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Build networks (must match training architecture)
    enc_img   = ImageEncoder(depth=32).to(device)
    embed_dim = enc_img.out_dim + 64

    # Get vec_dim from checkpoint indirectly via enc_vec weight shape
    enc_vec_w  = ckpt["models"]["enc_vec"]["net.0.weight"]
    vec_dim    = enc_vec_w.shape[1]
    act_dim_w  = ckpt["models"]["actor"]["mean_head.weight"]
    act_dim    = act_dim_w.shape[0]

    enc_vec  = VectorEncoder(vec_dim, 64).to(device)
    deter    = cfg["world_model"]["rssm_deter"]
    stoch    = cfg["world_model"]["rssm_stoch"]
    classes  = cfg["world_model"]["rssm_classes"]
    feat_dim = deter + stoch * classes

    rssm  = RSSM(deter, stoch, classes, embed_dim, act_dim).to(device)
    actor = Actor(feat_dim, act_dim, layers=3, units=256).to(device)

    enc_img.load_state_dict(ckpt["models"]["enc_img"])
    enc_vec.load_state_dict(ckpt["models"]["enc_vec"])
    rssm.load_state_dict(ckpt["models"]["rssm"])
    actor.load_state_dict(ckpt["models"]["actor"])

    enc_img.eval(); enc_vec.eval(); rssm.eval(); actor.eval()

    step    = ckpt.get("total_steps", 0)
    history = ckpt.get("ep_ret_history", [])
    print(f"  Trained for {step:,} steps")
    print(f"  Saved at: {ckpt.get('saved_at', 'unknown')}")

    return enc_img, enc_vec, rssm, actor, act_dim, history

# ─────────────────────────────────────────────────────────────────────────────
# Single episode rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(env, enc_img, enc_vec, rssm, actor,
                cfg, device, deterministic=True,
                record=False, frame_buffer=None):
    """
    Run one episode. Returns (total_reward, success, n_steps, frames).
    success = True if robosuite reports task completion.
    """
    obs_dict = env.reset()
    obs      = process_obs(obs_dict, cfg)
    h, s     = rssm.init_state(1, device)
    low, high = env.action_spec
    act_dim  = low.shape[0]

    total_reward = 0.0
    success      = False
    step         = 0
    frames       = []

    with torch.no_grad():
        while True:
            # Encode observation
            img_t = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
            vec_t = torch.from_numpy(obs["vector"]).unsqueeze(0).to(device)
            emb   = torch.cat([enc_img(img_t), enc_vec(vec_t)], -1)
            h, s  = rssm.obs_step(h, s, torch.zeros(1, act_dim, device=device), emb)

            # Get action from actor
            feat   = rssm.feat(h, s)
            action = actor.act(feat, deterministic=deterministic)
            action = action.squeeze(0).cpu().numpy()
            action = np.clip(action, low, high).astype(np.float64)

            # Step environment
            obs_dict, reward, done, info = env.step(action)
            obs           = process_obs(obs_dict, cfg)
            total_reward += reward
            step         += 1

            # Capture frame for video
            if record:
                frame = grab_frame(env, cfg, height=256, width=256)
                frames.append(frame)

            # Check task success
            if info.get("success", False) or info.get("task_complete", False):
                success = True

            if done:
                break

    return total_reward, success, step, frames

# ─────────────────────────────────────────────────────────────────────────────
# Video writer
# ─────────────────────────────────────────────────────────────────────────────

def save_video(frames, path, fps=20):
    """Save list of (H, W, 3) uint8 frames as MP4."""
    try:
        import imageio
        writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                    quality=8, macro_block_size=1)
        for f in frames:
            writer.append_data(f)
        writer.close()
        size_mb = Path(path).stat().st_size / 1e6
        print(f"  Video saved → {path}  ({len(frames)} frames, {size_mb:.1f} MB)")
    except Exception as e:
        print(f"  Video save failed: {e}")
        print(f"  Tip: pip install imageio imageio-ffmpeg")

# ─────────────────────────────────────────────────────────────────────────────
# Training curve plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curve(history, save_path):
    """Plot episode return history saved inside the checkpoint."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # no display needed
        import matplotlib.pyplot as plt

        history = list(history)
        if not history:
            print("  No training history in checkpoint — skipping plot")
            return

        x = np.arange(len(history))
        y = np.array(history, dtype=np.float32)

        # Smooth with rolling average
        window = min(20, len(y) // 4 + 1)
        smooth = np.convolve(y, np.ones(window)/window, mode="valid")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(x, y, alpha=0.3, color="#4C72B0", linewidth=0.8, label="Episode return")
        ax.plot(np.arange(len(smooth)) + window//2, smooth,
                color="#4C72B0", linewidth=2, label=f"Rolling mean ({window} ep)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return")
        ax.set_title("DreamerV3 Panda Pick-Place — Training Progress")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Training curve → {save_path}")
    except Exception as e:
        print(f"  Plot failed: {e}  (pip install matplotlib)")

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device(args.device)

    # ── Find checkpoint ───────────────────────────────────────────────────────
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = Path(cfg["logging"]["checkpoint_dir"]) / "latest.pt"

    if not ckpt_path.exists():
        print(f"ERROR: No checkpoint found at {ckpt_path}")
        print("  Train first:  MUJOCO_GL=glfw python scripts/train.py --config ...")
        sys.exit(1)

    # ── Load policy ───────────────────────────────────────────────────────────
    enc_img, enc_vec, rssm, actor, act_dim, history = \
        load_policy(ckpt_path, cfg, device)

    # ── Plot training curve ───────────────────────────────────────────────────
    log_dir = Path(cfg["logging"]["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)
    plot_path = log_dir / "training_curve.png"
    plot_training_curve(history, plot_path)

    # ── Build environment ─────────────────────────────────────────────────────
    print("\nBuilding environment...")
    env = make_env(cfg, record_video=args.render)

    # ── Output directory for videos ───────────────────────────────────────────
    render_dir = log_dir / "renders"
    render_dir.mkdir(exist_ok=True, parents=True)

    # ── Evaluation loop ───────────────────────────────────────────────────────
    print(f"\nEvaluating {args.episodes} episodes "
          f"({'deterministic' if args.deterministic else 'stochastic'} policy)...")
    print("-" * 60)

    all_rewards  = []
    all_successes = []
    all_steps    = []
    all_frames   = []

    for ep in range(args.episodes):
        record  = args.render and ep < args.render_episodes
        t_start = time.time()

        reward, success, steps, frames = run_episode(
            env, enc_img, enc_vec, rssm, actor, cfg, device,
            deterministic=args.deterministic,
            record=record,
        )

        all_rewards.append(reward)
        all_successes.append(success)
        all_steps.append(steps)
        if frames:
            all_frames.extend(frames)

        status = "✓ SUCCESS" if success else "✗ fail   "
        elapsed = time.time() - t_start
        print(f"  Ep {ep+1:>3}/{args.episodes}  {status}  "
              f"ret={reward:>8.3f}  steps={steps:>4}  "
              f"time={elapsed:.1f}s")

    # ── Save video ────────────────────────────────────────────────────────────
    if args.render and all_frames:
        ts         = datetime.now().strftime("%Y%m%d_%H%M")
        video_path = render_dir / f"rollout_{ts}.mp4"
        print(f"\nSaving video ({len(all_frames)} frames)...")
        save_video(all_frames, video_path, fps=cfg["env"]["control_freq"])

    # ── Summary report ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Episodes:       {args.episodes}")
    print(f"  Success rate:   {np.mean(all_successes)*100:.1f}%  "
          f"({sum(all_successes)}/{args.episodes})")
    print(f"  Mean return:    {np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}")
    print(f"  Max return:     {np.max(all_rewards):.3f}")
    print(f"  Min return:     {np.min(all_rewards):.3f}")
    print(f"  Mean ep length: {np.mean(all_steps):.0f} steps")
    print("=" * 60)

    # Interpret results
    sr = np.mean(all_successes) * 100
    print("\nInterpretation:")
    if sr == 0:
        print("  Policy hasn't learned to pick yet — train longer (need ~150k+ steps)")
    elif sr < 20:
        print("  Policy is reaching/grasping sometimes — good early signal")
    elif sr < 50:
        print("  Policy is learning! Grasping reliably, placing inconsistently")
    elif sr < 80:
        print("  Strong policy — consistent pick-and-place")
    else:
        print("  Excellent policy — near-expert performance")

    env.close()


if __name__ == "__main__":
    main()