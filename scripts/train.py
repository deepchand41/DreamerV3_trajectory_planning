"""
scripts/train.py  —  DreamerV3 Panda Pick-Place (M2 / MPS / CPU)

Auto-resume:  Automatically finds and loads the latest checkpoint on startup.
              No flags needed — just re-run the same command and it continues.

Timed saves:  Saves a checkpoint every 2 hours of wall-clock time,
              regardless of how many steps have been completed.
              Also saves on clean exit (Ctrl-C) so you never lose progress.

Usage:
    # First run — starts from scratch
    MUJOCO_GL=glfw python scripts/train.py --config configs/panda_pickplace_m2.yaml

    # Any subsequent run — auto-resumes from latest checkpoint
    MUJOCO_GL=glfw python scripts/train.py --config configs/panda_pickplace_m2.yaml

    # Force a fresh start (ignore existing checkpoints)
    MUJOCO_GL=glfw python scripts/train.py --config configs/panda_pickplace_m2.yaml --fresh
"""

import os, sys, argparse, yaml, time, signal
from pathlib import Path
from datetime import datetime, timedelta

_gl = os.environ.get("MUJOCO_GL", "glfw")
os.environ["MUJOCO_GL"] = _gl
os.environ["PYOPENGL_PLATFORM"] = _gl

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.device import get_device, device_info

# ─────────────────────────────────────────────────────────────────────────────
# Args + config
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",  default="configs/panda_pickplace_m2.yaml")
    p.add_argument("--device",  default="cpu", help="cpu | mps | cuda")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--fresh",   action="store_true",
                   help="Ignore existing checkpoints and start from scratch")
    p.add_argument("--save_interval_hours", type=float, default=2.0,
                   help="Save checkpoint every N hours (default: 2)")
    return p.parse_args()

def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)

def fmt_time(seconds):
    """Format seconds into human-readable string."""
    return str(timedelta(seconds=int(seconds)))

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(ckpt_dir: Path):
    """
    Scans checkpoint directory and returns path to the most recent checkpoint.
    Priority: latest.pt (always the newest) → step_XXXXXXXX.pt (by step number)
    Returns None if no checkpoints exist.
    """
    latest = ckpt_dir / "latest.pt"
    if latest.exists():
        return latest

    # Fallback: find highest-numbered step file
    step_files = sorted(ckpt_dir.glob("step_*.pt"))
    if step_files:
        return step_files[-1]

    return None


def save_checkpoint(ckpt_dir, models_dict, opts_dict,
                    total_steps, ep_ret_history, reason="timed"):
    """
    Saves a full checkpoint including:
      - All model weights
      - All optimiser states  ← critical for resuming correctly
      - Training step counter
      - Episode return history (for plotting)
      - Timestamp and reason for save
    """
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(exist_ok=True, parents=True)

    ckpt = {
        # ── Model weights ──────────────────────────────────────────────────
        "models": {k: v.state_dict() for k, v in models_dict.items()},

        # ── Optimiser states ───────────────────────────────────────────────
        # These contain momentum/Adam state — without them LR warmup restarts
        "optimisers": {k: v.state_dict() for k, v in opts_dict.items()},

        # ── Training state ─────────────────────────────────────────────────
        "total_steps":     total_steps,
        "ep_ret_history":  list(ep_ret_history),

        # ── Metadata ───────────────────────────────────────────────────────
        "saved_at":  datetime.now().isoformat(),
        "reason":    reason,   # "timed" | "exit" | "milestone"
    }

    # Write to a temp file first, then rename — prevents corruption if
    # the process is killed mid-write
    tmp  = ckpt_dir / "latest.tmp"
    dest = ckpt_dir / "latest.pt"
    torch.save(ckpt, tmp)
    tmp.rename(dest)

    # Also save a timestamped copy so you can roll back
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    named = ckpt_dir / f"step_{total_steps:08d}_{ts}.pt"
    torch.save(ckpt, named)

    size_mb = dest.stat().st_size / 1e6
    print(f"\n  ✓ Checkpoint saved  [{reason}]")
    print(f"    Steps: {total_steps:,}")
    print(f"    File:  {named.name}  ({size_mb:.1f} MB)")
    print(f"    Time:  {datetime.now().strftime('%H:%M:%S')}\n")

    return named


def load_checkpoint(path, models_dict, opts_dict, device):
    """
    Loads weights AND optimiser states from a checkpoint.
    Returns the step to resume from.
    """
    print(f"\n  ↩  Resuming from: {path}")
    ckpt = torch.load(path, map_location=device)

    # Load model weights
    for name, model in models_dict.items():
        if name in ckpt["models"]:
            model.load_state_dict(ckpt["models"][name])
            print(f"     loaded: {name}")
        else:
            print(f"     [warn] {name} not found in checkpoint — using fresh weights")

    # Load optimiser states (restores Adam momentum)
    for name, opt in opts_dict.items():
        if name in ckpt.get("optimisers", {}):
            opt.load_state_dict(ckpt["optimisers"][name])
            print(f"     loaded optimiser: {name}")

    step    = ckpt.get("total_steps", 0)
    history = deque(ckpt.get("ep_ret_history", []), maxlen=200)

    saved_at = ckpt.get("saved_at", "unknown")
    reason   = ckpt.get("reason",   "unknown")
    print(f"     Resuming from step {step:,}  (saved: {saved_at}, reason: {reason})\n")

    return step, history

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

def make_raw_env(cfg):
    import robosuite as suite
    from robosuite.controllers import load_controller_config
    return suite.make(
        env_name=cfg["env"]["task"],
        robots=cfg["env"]["robot"],
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cfg["env"]["camera"],
        camera_heights=cfg["env"]["image_size"],
        camera_widths=cfg["env"]["image_size"],
        reward_shaping=cfg["env"]["reward_shaping"],
        control_freq=cfg["env"]["control_freq"],
        horizon=cfg["env"]["horizon"],
        ignore_done=False,
        controller_configs=load_controller_config(
            default_controller=cfg["env"]["controller"]
        ),
    )


class RobosuiteWrapper:
    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self._low, self._high = env.action_spec
        self.act_dim = self._low.shape[0]

    def sample_action(self):
        return np.random.uniform(self._low, self._high).astype(np.float32)

    def reset(self):
        return self._process(self.env.reset())

    def step(self, action):
        action = np.clip(action.astype(np.float64), self._low, self._high)
        obs_dict, reward, done, info = self.env.step(action)
        return self._process(obs_dict), float(reward), bool(done), info

    def _process(self, obs_dict):
        cam_key = self.cfg["env"]["camera"] + "_image"
        img = obs_dict[cam_key]
        img = np.ascontiguousarray(np.flipud(img)).transpose(2, 0, 1)
        parts = []
        for k in self.cfg["env"]["obs_keys"]:
            if k in obs_dict:
                parts.append(np.asarray(obs_dict[k], dtype=np.float32).flatten())
        vec = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        return {"image": img.copy(), "vector": vec}

    def close(self):
        self.env.close()

# ─────────────────────────────────────────────────────────────────────────────
# Replay buffer
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    def __init__(self, capacity, seq_len):
        self.buf     = deque(maxlen=capacity)
        self.seq_len = seq_len

    def add(self, img, vec, act, rew, done):
        self.buf.append((img.copy(), vec.copy(),
                         np.asarray(act, np.float32),
                         float(rew), bool(done)))

    def ready(self, batch_size):
        return len(self.buf) >= batch_size * self.seq_len + 1

    def sample(self, batch_size, device):
        buf = list(self.buf)
        N   = len(buf) - self.seq_len
        idx = np.random.randint(0, N, batch_size)
        def g(field):
            return np.stack([[buf[i+t][field] for t in range(self.seq_len)]
                             for i in idx])
        imgs  = torch.from_numpy(g(0)).to(device)
        vecs  = torch.from_numpy(g(1)).to(device)
        acts  = torch.from_numpy(g(2)).to(device)
        rews  = torch.from_numpy(g(3).astype(np.float32)).to(device)
        dones = torch.from_numpy(g(4).astype(np.float32)).to(device)
        return imgs, vecs, acts, rews, dones

# ─────────────────────────────────────────────────────────────────────────────
# Networks
# ─────────────────────────────────────────────────────────────────────────────

def safe(x, name=""):
    if torch.isnan(x).any() or torch.isinf(x).any():
        if name:
            print(f"  [NaN guard] {name} — replacing with 0")
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
    return x


class ImageEncoder(nn.Module):
    def __init__(self, depth=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, depth,      4, 2, 0), nn.BatchNorm2d(depth),    nn.SiLU(),
            nn.Conv2d(depth, depth*2, 4, 2, 0), nn.BatchNorm2d(depth*2), nn.SiLU(),
            nn.Conv2d(depth*2,depth*4,4, 2, 0), nn.BatchNorm2d(depth*4), nn.SiLU(),
            nn.Conv2d(depth*4,depth*8,4, 2, 0), nn.BatchNorm2d(depth*8), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 3, 64, 64)).shape[1]

    def forward(self, x):
        return safe(self.net(x.float() / 255.0 - 0.5), "img_enc")


class VectorEncoder(nn.Module):
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return safe(self.net(x), "vec_enc")


class RSSM(nn.Module):
    def __init__(self, deter, stoch, classes, embed_dim, action_dim):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.lat = stoch * classes
        self.gru_in   = nn.Linear(self.lat + action_dim, deter)
        self.gru_norm = nn.LayerNorm(deter)
        self.gru      = nn.GRUCell(deter, deter)
        H = min(deter, 512)
        self.prior = nn.Sequential(
            nn.Linear(deter, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, stoch * classes))
        self.post = nn.Sequential(
            nn.Linear(deter + embed_dim, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, stoch * classes))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def init_state(self, B, device):
        return (torch.zeros(B, self.deter, device=device),
                torch.zeros(B, self.stoch, self.classes, device=device))

    def feat(self, h, s):
        return torch.cat([h, s.view(s.shape[0], -1)], -1)

    def _gumbel(self, logits):
        logits = safe(logits).clamp(-10, 10)
        dev = logits.device
        return F.gumbel_softmax(logits.cpu(), tau=1.0, hard=True, dim=-1).to(dev)

    def obs_step(self, h, s, a, e):
        x  = self.gru_norm(self.gru_in(
             torch.cat([s.view(s.shape[0],-1).clamp(-5,5), a.clamp(-5,5)], -1)))
        h2 = safe(self.gru(safe(x), h)).clamp(-20, 20)
        pri = safe(self.prior(h2).view(-1, self.stoch, self.classes))
        pos = safe(self.post(torch.cat([h2, safe(e)], -1)).view(-1, self.stoch, self.classes))
        return h2, self._gumbel(pos), pri, pos

    def img_step(self, h, s, a):
        x  = self.gru_norm(self.gru_in(
             torch.cat([s.view(s.shape[0],-1).clamp(-5,5), a.clamp(-5,5)], -1)))
        h2 = safe(self.gru(safe(x), h)).clamp(-20, 20)
        pri = safe(self.prior(h2).view(-1, self.stoch, self.classes))
        return h2, self._gumbel(pri)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, layers=2, units=256):
        super().__init__()
        net = [nn.Linear(in_dim, units), nn.LayerNorm(units), nn.SiLU()]
        for _ in range(layers-1):
            net += [nn.Linear(units, units), nn.LayerNorm(units), nn.SiLU()]
        net += [nn.Linear(units, out_dim)]
        self.net = nn.Sequential(*net)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.1)
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, z):
        h    = self.trunk(safe(z, "actor_in"))
        mean = safe(self.mean_head(h), "actor_mean").clamp(-10, 10)
        std  = safe(F.softplus(self.std_head(h)), "actor_std").clamp(self.min_std, 10.0)
        return torch.distributions.Independent(
            torch.distributions.Normal(mean, std), 1)


class Critic(nn.Module):
    def __init__(self, feat_dim, layers=3, units=256):
        super().__init__()
        net = [nn.Linear(feat_dim, units), nn.LayerNorm(units), nn.SiLU()]
        for _ in range(layers-1):
            net += [nn.Linear(units, units), nn.LayerNorm(units), nn.SiLU()]
        net += [nn.Linear(units, 1)]
        self.net = nn.Sequential(*net)
    def forward(self, z): return self.net(safe(z, "critic_in")).squeeze(-1)


def symlog(x): return torch.sign(x) * torch.log1p(x.abs())
def symexp(x): return torch.sign(x) * (x.abs().exp() - 1)


def categorical_kl(post_logits, prior_logits):
    log_post  = F.log_softmax(post_logits,  dim=-1)
    log_prior = F.log_softmax(prior_logits, dim=-1)
    kl = (log_post.exp() * (log_post - log_prior)).sum(-1).sum(-1)
    return kl


def lambda_returns(rewards, values, continues, lam=0.95):
    H   = rewards.shape[1]
    ret = torch.zeros_like(rewards)
    last = values[:, -1]
    for t in reversed(range(H)):
        last = rewards[:, t] + continues[:, t] * ((1-lam)*values[:, t] + lam*last)
        ret[:, t] = last
    return ret

# ─────────────────────────────────────────────────────────────────────────────
# Training step
# ─────────────────────────────────────────────────────────────────────────────

def train_step(batch, models, opts, cfg, device):
    enc_img, enc_vec, rssm, rew_h, cont_h, actor, critic = models
    opt_wm, opt_actor, opt_critic = opts

    imgs, vecs, acts, rews, dones = batch
    B, T = imgs.shape[:2]

    emb_img = enc_img(imgs.view(B*T, *imgs.shape[2:]))
    emb_vec = enc_vec(vecs.view(B*T, *vecs.shape[2:]))
    embed   = torch.cat([emb_img, emb_vec], -1).view(B, T, -1)

    h, s = rssm.init_state(B, device)
    feats, pris, posts = [], [], []
    for t in range(T):
        if t > 0:
            r = dones[:, t-1].unsqueeze(-1)
            h = h * (1 - r); s = s * (1 - r.unsqueeze(-1))
        h, s, pri, pos = rssm.obs_step(h, s, acts[:, t], embed[:, t])
        feats.append(rssm.feat(h, s))
        pris.append(pri); posts.append(pos)

    feats = torch.stack(feats, 1)
    pris  = torch.stack(pris,  1)
    posts = torch.stack(posts, 1)
    feat_flat = feats.view(B*T, -1)

    kl_loss   = categorical_kl(posts, pris).clamp(min=1.0).mean()
    rew_loss  = F.mse_loss(rew_h(feat_flat).squeeze(-1), symlog(rews.view(B*T)))
    cont_loss = F.binary_cross_entropy_with_logits(
        cont_h(feat_flat).squeeze(-1), (1 - dones).view(B*T))
    wm_loss   = kl_loss + rew_loss + cont_loss

    if torch.isnan(wm_loss):
        return None

    opt_wm.zero_grad(); wm_loss.backward()
    wm_params = (list(enc_img.parameters()) + list(enc_vec.parameters()) +
                 list(rssm.parameters())    + list(rew_h.parameters()) +
                 list(cont_h.parameters()))
    nn.utils.clip_grad_norm_(wm_params, 10.0)
    opt_wm.step()

    H      = cfg["actor_critic"]["imagination_horizon"]
    starts = feats.view(B*T, -1).detach()
    idx    = torch.randperm(starts.shape[0])[:min(128, starts.shape[0])]
    starts = starts[idx]; Bs = starts.shape[0]
    dh     = rssm.deter
    hi     = starts[:, :dh]
    si     = starts[:, dh:].view(Bs, rssm.stoch, rssm.classes)

    im_feats, im_rews, im_conts = [], [], []
    for _ in range(H):
        fi = rssm.feat(hi, si)
        ai = actor(fi).rsample().clamp(-1, 1)
        hi, si = rssm.img_step(hi, si, ai)
        im_feats.append(fi)
        im_rews.append(symexp(safe(rew_h(fi).squeeze(-1))).detach())
        im_conts.append(torch.sigmoid(safe(cont_h(fi).squeeze(-1))).detach())

    im_feats = torch.stack(im_feats, 1)
    im_rews  = torch.stack(im_rews,  1)
    im_conts = torch.stack(im_conts, 1)

    with torch.no_grad():
        vals = critic(im_feats.view(Bs*H, -1)).view(Bs, H)
    returns  = lambda_returns(im_rews, vals, im_conts, cfg["actor_critic"]["lambda_"])
    ret_flat = returns.view(Bs*H).detach()

    dist2      = actor(im_feats.view(Bs*H, -1))
    norm_ret   = (ret_flat - ret_flat.mean()) / (ret_flat.std() + 1e-8)
    actor_loss = (-norm_ret - cfg["actor_critic"]["actor_entropy"] * dist2.entropy()).mean()

    if not torch.isnan(actor_loss):
        opt_actor.zero_grad(); actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
        opt_actor.step()

    crit_loss = F.mse_loss(critic(im_feats.detach().view(Bs*H, -1)), ret_flat)
    if not torch.isnan(crit_loss):
        opt_critic.zero_grad(); crit_loss.backward()
        nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
        opt_critic.step()

    return {
        "wm":    wm_loss.item(),
        "kl":    kl_loss.item(),
        "rew":   rew_loss.item(),
        "actor": actor_loss.item() if not torch.isnan(actor_loss) else float("nan"),
        "crit":  crit_loss.item(),
    }

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = get_device(args.device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    log_dir  = Path(cfg["logging"]["log_dir"])
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    log_dir.mkdir(exist_ok=True,  parents=True)

    # ── Build environment ─────────────────────────────────────────────────────
    print("Building environment...")
    raw     = make_raw_env(cfg)
    env     = RobosuiteWrapper(raw, cfg)
    act_dim = env.act_dim
    obs0    = env.reset()
    vec_dim = obs0["vector"].shape[0]

    # ── Build networks ────────────────────────────────────────────────────────
    enc_img  = ImageEncoder(depth=32).to(device)
    enc_vec  = VectorEncoder(vec_dim, 64).to(device)
    embed_dim = enc_img.out_dim + 64
    deter    = cfg["world_model"]["rssm_deter"]
    stoch    = cfg["world_model"]["rssm_stoch"]
    classes  = cfg["world_model"]["rssm_classes"]
    feat_dim = deter + stoch * classes

    rssm   = RSSM(deter, stoch, classes, embed_dim, act_dim).to(device)
    rew_h  = MLP(feat_dim, 1, layers=2, units=256).to(device)
    cont_h = MLP(feat_dim, 1, layers=2, units=256).to(device)
    actor  = Actor(feat_dim, act_dim, layers=3, units=256).to(device)
    critic = Critic(feat_dim, layers=3, units=256).to(device)

    # ── Optimisers ────────────────────────────────────────────────────────────
    wm_params = (list(enc_img.parameters()) + list(enc_vec.parameters()) +
                 list(rssm.parameters())    + list(rew_h.parameters()) +
                 list(cont_h.parameters()))
    opt_wm     = torch.optim.Adam(wm_params,          lr=1e-4,  eps=1e-8)
    opt_actor  = torch.optim.Adam(actor.parameters(),  lr=3e-5, eps=1e-5)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=3e-5, eps=1e-5)

    # Named dicts for clean checkpoint save/load
    models_dict = {"enc_img": enc_img, "enc_vec": enc_vec, "rssm": rssm,
                   "rew_h": rew_h, "cont_h": cont_h, "actor": actor, "critic": critic}
    opts_dict   = {"opt_wm": opt_wm, "opt_actor": opt_actor, "opt_critic": opt_critic}
    models      = (enc_img, enc_vec, rssm, rew_h, cont_h, actor, critic)
    opts        = (opt_wm, opt_actor, opt_critic)

    # ── Auto-resume ───────────────────────────────────────────────────────────
    total_steps    = 0
    ep_ret_history = deque(maxlen=200)

    if not args.fresh:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest:
            total_steps, ep_ret_history = load_checkpoint(
                latest, models_dict, opts_dict, device)
        else:
            print("  No checkpoint found — starting from scratch.\n")
    else:
        print("  --fresh flag set — ignoring any existing checkpoints.\n")

    # ── Print summary ─────────────────────────────────────────────────────────
    total_params = sum(p.numel() for m in models for p in m.parameters())
    print(f"Device:      {device_info(device)}")
    print(f"Parameters:  {total_params:,}")
    print(f"Resume step: {total_steps:,} / {cfg['training']['total_steps']:,}")
    print(f"Save every:  {args.save_interval_hours} hours")
    print(f"Checkpoint:  {ckpt_dir}\n")

    # ── Replay buffer ─────────────────────────────────────────────────────────
    buf = ReplayBuffer(capacity=cfg["replay"]["capacity"],
                       seq_len=cfg["replay"]["batch_length"])
    PREFILL = cfg["training"]["prefill_steps"]
    BATCH   = cfg["replay"]["batch_size"]

    # If resuming past prefill, we still need to refill the buffer with random
    # transitions before training can continue (buffer isn't saved — too large)
    if total_steps >= PREFILL:
        print(f"Refilling replay buffer with {PREFILL} random steps "
              f"(buffer not saved in checkpoint)...")
        obs = env.reset()
        for _ in range(PREFILL):
            a = env.sample_action()
            next_obs, r, done, _ = env.step(a)
            buf.add(obs["image"], obs["vector"], a, r, done)
            obs = env.reset() if done else next_obs
        print(f"  Buffer ready ({len(buf.buf):,} transitions)\n")
        obs = env.reset()
    else:
        obs = obs0

    # ── Timing state ─────────────────────────────────────────────────────────
    SAVE_INTERVAL_SEC = args.save_interval_hours * 3600
    last_save_time    = time.time()   # reset timer after resume
    session_start     = time.time()
    h_state, s_state  = rssm.init_state(1, device)
    ep_ret, ep_steps  = 0.0, 0
    nan_skips         = 0
    losses            = {}

    # ── Graceful exit on Ctrl-C ───────────────────────────────────────────────
    interrupted = False
    def _handle_sigint(sig, frame):
        nonlocal interrupted
        print("\n\n  Ctrl-C received — saving checkpoint before exit...")
        interrupted = True
    signal.signal(signal.SIGINT, _handle_sigint)

    if total_steps < PREFILL:
        print(f"Prefilling {PREFILL} random steps...")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while total_steps < cfg["training"]["total_steps"] and not interrupted:

        # Act
        with torch.no_grad():
            img_t = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
            vec_t = torch.from_numpy(obs["vector"]).unsqueeze(0).to(device)
            emb   = torch.cat([enc_img(img_t), enc_vec(vec_t)], -1)
            h_state, s_state, *_ = rssm.obs_step(
                h_state, s_state,
                torch.zeros(1, act_dim, device=device), emb)

            if total_steps < PREFILL:
                action = env.sample_action()
            else:
                feat   = rssm.feat(h_state, s_state)
                action = actor(feat).sample().squeeze(0).cpu().numpy()
                action = np.clip(action, -1.0, 1.0).astype(np.float32)

        # Step
        next_obs, reward, done, _ = env.step(action)
        buf.add(obs["image"], obs["vector"], action, reward, done)
        ep_ret   += reward
        ep_steps += 1
        obs       = next_obs
        total_steps += 1

        if done:
            ep_ret_history.append(ep_ret)
            recent_mean = np.mean(list(ep_ret_history)[-20:]) if ep_ret_history else 0
            elapsed     = time.time() - session_start
            print(f"[{total_steps:>8,}]  ret={ep_ret:.3f}  "
                  f"mean20={recent_mean:.3f}  "
                  f"elapsed={fmt_time(elapsed)}")
            obs          = env.reset()
            h_state, s_state = rssm.init_state(1, device)
            ep_ret   = 0.0
            ep_steps = 0

        # Train
        if total_steps >= PREFILL and buf.ready(BATCH):
            for _ in range(cfg["training"]["train_ratio"]):
                result = train_step(buf.sample(BATCH, device), models, opts, cfg, device)
                if result is None:
                    nan_skips += 1
                else:
                    losses = result

            if total_steps % cfg["training"]["log_every"] == 0 and losses:
                elapsed = time.time() - session_start
                next_save_in = SAVE_INTERVAL_SEC - (time.time() - last_save_time)
                print(f"  loss  wm={losses['wm']:.3f}  kl={losses['kl']:.3f}  "
                      f"rew={losses['rew']:.4f}  actor={losses['actor']:.3f}  "
                      f"nan={nan_skips}  "
                      f"next_save={fmt_time(max(0, next_save_in))}")

        # ── Timed checkpoint (every N hours) ──────────────────────────────────
        now = time.time()
        if now - last_save_time >= SAVE_INTERVAL_SEC:
            save_checkpoint(ckpt_dir, models_dict, opts_dict,
                            total_steps, ep_ret_history, reason="timed_2h")
            last_save_time = now

    # ── Final save (on completion or Ctrl-C) ──────────────────────────────────
    reason = "interrupted" if interrupted else "completed"
    save_checkpoint(ckpt_dir, models_dict, opts_dict,
                    total_steps, ep_ret_history, reason=reason)

    env.close()
    elapsed = time.time() - session_start
    print(f"\nDone. Total steps: {total_steps:,}  Session time: {fmt_time(elapsed)}")


if __name__ == "__main__":
    main()