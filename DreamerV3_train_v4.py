"""
scripts/train.py — DreamerV3 (official-aligned) for Robosuite Panda PickPlaceCan

Faithful PyTorch implementation of Hafner et al. 2023 (arXiv:2301.04104),
adapted for robosuite manipulation.

═══════════════════════════════════════════════════════════════════════════════
REVISION 5 — stability, after the run diverged at ~84k steps
═══════════════════════════════════════════════════════════════════════════════

WHAT HAPPENED AT 84k
    ~84k  world model blew up
            world_model/reconstruction   ~10  ->  ~2000   (200x)
            world_model/vector_recon     0.02 ->  4.1     (200x)
            world_model/recon_std_ratio  ~1   ->  135
    ~84k  actor followed
            actor/entropy  -> -18 then FLAT  (std pinned at min_std=0.1)
            actor/loss     -> -7 then frozen at -2.5
            actor/grad_norm-> ~0
    ~85k  everything froze
            returns/scale pinned at the 1.0 floor; returns/mean flat
            train/nan_skips  0 -> 100,000+  climbing LINEARLY

  The linear climb is the tell. A transient instability spikes and settles.
  A linear climb means every batch is failing identically, which happens once
  inf/NaN is resident in the RSSM or encoder PARAMETERS. Revision 4's guard

        out = train_step(...)
        if out is None:  nan_skips += 1

  detected the non-finite loss and skipped the batch, but never removed the
  corruption, so the same poisoned weights were reused forever. At
  train_ratio=32 that was ~3,100 env steps of zero learning.

REVISION 5 CHANGES
    FIX 8   snapshot / rewind. A clean copy of every module AND optimiser is
            kept, refreshed every `snapshot_every` gradient steps. After
            `nan_patience` consecutive failed batches the run rewinds to it
            and halves all learning rates. Restoring weights WITHOUT the
            optimiser state is the usual way this silently fails: Adam's
            exp_avg / exp_avg_sq are poisoned by a NaN gradient too, and
            keeping them re-injects the corruption on the very next step.
    FIX 9   tripwire on world_model/reconstruction and recon_std_ratio.
            recon_std_ratio reached 135 during the event and it MOVES BEFORE
            THE LOSS DOES, so it is the earliest available warning. Tripping
            at 5 stops the run while the weights are still clean.
    FIX 10  replay-buffer sanity scan at startup. One non-finite transition
            will re-trigger the blow-up whenever it happens to be sampled,
            and it will look spontaneous.
    FIX 11  --checkpoint lets you resume from a NAMED file. This is required
            here: checkpoints/latest.pt was overwritten with post-divergence
            weights, so the 80k checkpoint must be pulled from the W&B
            artifact store and passed explicitly.
    FIX 12  FIX 7 (actor Adam reset on resume) is now OPT-IN via
            --reset_actor_opt. It was correct for the rev2 -> rev3 transition
            because max_std changed underneath the optimiser. Resuming rev4 ->
            rev5 changes no actor hyperparameter, so silently discarding the
            actor's moments would throw away 25k steps of adaptation.

  CONFIG changes that go with this revision (in the YAML, not here):
      training.mixed_precision          true  -> false   <- most likely cause
      world_model.grad_clip             1000  -> 50
      actor_critic.actor_grad_clip      100   -> 5
      actor_critic.critic_grad_clip     100   -> 5
      actor_critic.actor_min_std        0.1   -> 0.2
      training.train_ratio              32    -> 16
      training.total_steps                    -> 88_000  (hard stop)
      + new `stability:` block

  WHY fp16 IS THE PRIME SUSPECT: fp16 maxes out at 65504. A reconstruction
  term that reached ~2000 had intermediate activations far above that. One
  overflow to inf in the forward pass writes NaN into the weights on the
  backward pass, and from then on every batch fails identically -- exactly
  the observed signature. The T4 is Turing, so bf16 is unavailable; fp32 is
  the only safe option. Costs ~30-40% throughput, which train_ratio 16 repays.

  actor_max_std STAYS AT 1.0. Do not lower it for stability. At a gripper
  mean of -0.52, sigma=1.0 gives 14.2% closure per step and sigma=0.5 gives
  1.6% -- that change is what killed grasping in revision 3.

═══════════════════════════════════════════════════════════════════════════════
REVISION 3 — actor fixes, after the world model was measured healthy
═══════════════════════════════════════════════════════════════════════════════

REVISION 1 (posterior collapse) — fixed in rev 2, retained here for context
    world_model/loss  flat at 0.645 = 0.600 (KL free-bits floor) + 0.045
    world_model/kl    0.6 over 32 latents = 0.019 nats/latent (~zero)
    world_model/grad_norm  0.01 and flat -> the world model had stopped learning
  Cause: F.mse_loss(rec, tgt) averages over all 3x64x64 = 12,288 pixel dims,
  making reconstruction ~12,000x smaller than the paper's. It contributed
  0.005 against a KL of 0.600, so the cheapest minimum was for the posterior
  to stop depending on `embed`. Free bits could not prevent this: it removes
  the pressure to SHRINK the KL, it cannot create pressure to GROW it.
    FIX 1  recon sums over pixel dims, averages over batch (paper scale)
    FIX 2  lambda_returns applies gamma (cont_head had collapsed to ~1.0, so
           the agent was running gamma=1.0 and values drifted to ~3)
    FIX 3  world-model grad_clip 100 -> 1000 (paper)
    FIX 4  actor_entropy 1e-3 -> 3e-4 (paper)
    FIX 5  vector decoder head (the paper decodes ALL observation modalities)

REVISION 2 (measured) — world model healthy, actor parked
    recon 60 -> ~5 falling; vecrec 2.1 -> 0.05; recon_std_ratio ~1.0
    32/32 latents active, changing category on 35-62% of steps
    min_distance 0.198 -> 0.140 monotone (the first real policy signal)
  But: actor/entropy pinned at exactly 9.93 = 7 * 0.5*log(2*pi*e*1.0^2),
  the analytic maximum at max_std=1.0, dead flat for 5k steps.

DIAGNOSIS (check_action_response.py)
  The obvious hypothesis -- that imagination is action-blind -- was FALSE:
      |d(return)/d(action)| = 0.82   vs   |d(return)/d(latent)| = 1.05
      ratio 0.78, i.e. the same order of magnitude
  The actor receives a perfectly usable action gradient. The real constraint
  is PRECISION:
      open-loop eef_pos error at H=15   0.125 m
      task reach threshold              0.050 m
  The world model is 2.5x less precise than the task requires. Coarse
  approach improves (direction is enough); grasping cannot (it needs
  centimetre precision). Error concentrates in joint_pos (0.60 rad mean)
  because the 30-dim proprioception vector carries under 1% of the world
  model's prediction loss, while the image term -- dominated by static
  table, bins and background -- carries the rest.

REVISION 3 CHANGES
    FIX 6  entropy estimated as -log p(a) of the SQUASHED distribution
           instead of the pre-squash Gaussian entropy. The old quantity rises
           without bound in std, so the bonus always drove std to the clamp.
    FIX 7  actor Adam moments discarded on resume; they were accumulated with
           std pinned at the old clamp.  (Now opt-in -- see FIX 12.)

═══════════════════════════════════════════════════════════════════════════════

Official DreamerV3 components implemented here:
  - symlog observation transform (vector obs) and symlog decode targets
  - symexp two-hot distributional loss (255 bins) for reward head AND critic
  - KL balancing: dyn (beta=0.5) + rep (beta=0.1) with free bits at 1.0 nat
  - 1% unimix on all categorical distributions
  - percentile return normalisation: S = EMA(Per95 - Per5), divide by max(1, S)
  - slow critic (EMA target, decay 0.98) + slow-critic regularisation
  - zero-initialised output layers on reward head, critic, and actor mean
  - continuation-weighted imagination losses
  - RMSNorm-style normalisation in the conv stacks

Known deliberate deviations from the official JAX implementation:
  - Adam instead of LaProp; norm-based clipping instead of adaptive (AGC)
  - standard nn.GRUCell instead of the block-diagonal GRU
  - LayerNorm in MLPs, RMSNorm only in conv stacks
  - actor_grad='dynamics' (reparameterised) rather than REINFORCE

Usage:
    # revision 5 resume from the recovered 80k checkpoint
    python scripts/train.py \
        --config configs/panda_pickplace_stable.yaml \
        --checkpoint checkpoints/grasping_80k.pt

    # fresh run
    python scripts/train.py --config configs/panda_pickplace_colab.yaml --fresh
"""

import os, sys, csv, math, time, copy, yaml, signal, argparse
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
    WANDB_OK = True
except ImportError:
    WANDB_OK = False

sys.path.insert(0, "/content/dreamerv3_panda")
try:
    from utils.device import get_device, device_info
except Exception:
    def get_device(prefer="auto"):
        if prefer == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(prefer)
    def device_info(d):
        if d.type == "cuda":
            n = torch.cuda.get_device_name(0)
            m = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"CUDA - {n} ({m:.1f} GB)"
        return "CPU"


# ══════════════════════════════════════════════════════════════════════════════
# Args / config
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/panda_pickplace_colab.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--max_minutes", type=float, default=0)
    p.add_argument("--no_save_buffer", action="store_true")
    p.add_argument("--no_wandb", action="store_true")
    # FIX 11 — resume from a NAMED checkpoint. Required after the 84k
    # divergence, because checkpoints/latest.pt was overwritten with
    # post-divergence weights and the good 80k state lives in W&B artifacts.
    p.add_argument("--checkpoint", default=None,
                   help="explicit checkpoint path; overrides latest.pt discovery")
    # FIX 12 — the old FIX 7 behaviour, now opt-in. Only use it when an
    # actor hyperparameter (max_std, min_std, entropy scale) changed relative
    # to the checkpoint being resumed.
    p.add_argument("--reset_actor_opt", action="store_true",
                   help="discard the actor's Adam moments on resume (FIX 7)")
    # Colab-safe: ignore notebook argv noise
    known, _ = p.parse_known_args()
    return known


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def cfgget(cfg, section, key, default):
    return cfg.get(section, {}).get(key, default)


def fmt_time(s):
    return str(timedelta(seconds=int(s)))


def safe(x):
    """Replace non-finite values. Cheap insurance against fp16 edge cases."""
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
    return x


# ══════════════════════════════════════════════════════════════════════════════
# Official DreamerV3 math: symlog, two-hot, unimix, return normalisation
# ══════════════════════════════════════════════════════════════════════════════

def symlog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    return torch.sign(x) * (x.abs().exp() - 1.0)


class TwoHot:
    """
    Official DreamerV3 discrete regression head (Section 3, "symexp twohot").

    A scalar target y is encoded as a soft distribution over K=255 bins whose
    positions are symexp-spaced in [-20, 20] symlog space. The network outputs
    logits over bins; the loss is cross-entropy. Prediction is the expected
    bin value, then symexp'd back to real reward/return scale.

    Why this matters for PickPlaceCan:
      Robosuite rewards jump from ~0.0002 (far from can) to 0.0875 (grasping),
      a ~400x range. With plain MSE the rare large reward dominates the
      gradient. Two-hot cross-entropy makes the gradient magnitude independent
      of the target magnitude.
    """

    def __init__(self, num_bins=255, lo=-20.0, hi=20.0, device="cpu"):
        self.num_bins = num_bins
        # Bins are evenly spaced in SYMLOG space; the values they represent
        # are symexp of those positions.
        self.sym_pos = torch.linspace(lo, hi, num_bins, device=device)

    def to(self, device):
        self.sym_pos = self.sym_pos.to(device)
        return self

    def encode(self, y):
        """
        y: (N,) real-valued targets (raw reward / return scale)
        returns: (N, K) two-hot soft targets in symlog space
        """
        ys = symlog(y).clamp(self.sym_pos[0], self.sym_pos[-1])   # (N,)
        pos = self.sym_pos                                        # (K,)
        # index of the bin immediately at/below ys
        below = (pos.unsqueeze(0) <= ys.unsqueeze(1)).sum(-1) - 1
        below = below.clamp(0, self.num_bins - 2)
        above = below + 1
        lo = pos[below]
        hi = pos[above]
        w_hi = (ys - lo) / (hi - lo + 1e-8)
        w_lo = 1.0 - w_hi
        out = torch.zeros(y.shape[0], self.num_bins, device=y.device, dtype=torch.float32)
        out.scatter_(1, below.unsqueeze(1), w_lo.unsqueeze(1).float())
        out.scatter_(1, above.unsqueeze(1), w_hi.unsqueeze(1).float())
        return out

    def loss(self, logits, y):
        """Cross-entropy between predicted bin distribution and two-hot target."""
        tgt = self.encode(y.detach())
        logp = F.log_softmax(logits.float(), dim=-1)
        return -(tgt * logp).sum(-1)          # (N,)  -- caller reduces

    def mean(self, logits):
        """Expected value, mapped back to real scale via symexp."""
        probs = torch.softmax(logits.float(), dim=-1)
        sym_mean = (probs * self.sym_pos.unsqueeze(0)).sum(-1)
        return symexp(sym_mean)


def unimix_logits(logits, ratio=0.01):
    """
    Official 1% uniform mixture. Returns *logits* (not log-probs) so the result
    can still be fed to gumbel_softmax and to log_softmax consistently.

    Guarantees no category has probability exactly 0, so the KL can never
    become infinite and gradients never vanish for unused categories.
    """
    probs = torch.softmax(logits, dim=-1)
    uniform = torch.ones_like(probs) / probs.shape[-1]
    mixed = (1.0 - ratio) * probs + ratio * uniform
    return torch.log(mixed + 1e-10)


def kl_categorical(post_logits, prior_logits):
    """KL(post || prior) summed over classes then over the stoch dimension."""
    lp = F.log_softmax(post_logits, -1)
    lq = F.log_softmax(prior_logits, -1)
    return (lp.exp() * (lp - lq)).sum(-1).sum(-1)      # (B, T)


def categorical_entropy(logits):
    """Mean per-latent entropy in nats. Max = log(classes) = 3.466 for 32."""
    lp = F.log_softmax(logits, -1)
    return -(lp.exp() * lp).sum(-1).mean()


def kl_balanced(post_logits, prior_logits,
                free_nats=1.0, beta_dyn=0.5, beta_rep=0.1, unimix=0.01):
    """
    Official DreamerV3 KL balancing (paper Eq. 5):

        L_dyn = max(free, KL( sg(post) || prior ))     * beta_dyn
        L_rep = max(free, KL( post || sg(prior) ))     * beta_rep

    L_dyn trains the *prior* (sequence model) to predict the posterior.
    L_rep trains the *posterior* (encoder) to be predictable.

    free_nats=1.0 is the "free bits" floor, clamped ELEMENT-WISE before the
    mean (clamping after the mean would defeat the purpose). It stops the model
    being pushed to compress further once the prior is already good enough.

    NOTE: free bits is a one-sided guard. It removes the incentive to shrink
    the KL; it cannot stop the posterior collapsing if some other term (here,
    reconstruction) is too weakly scaled to demand observation information.
    That is precisely what happened in revision 1 -- see the header.

    Returns (total_loss, kl_value_for_logging).
    """
    post_l = unimix_logits(post_logits, unimix)
    prior_l = unimix_logits(prior_logits, unimix)

    kl_dyn = kl_categorical(post_l.detach(), prior_l)          # grads -> prior
    kl_rep = kl_categorical(post_l, prior_l.detach())          # grads -> post

    kl_dyn = kl_dyn.clamp(min=free_nats).mean()
    kl_rep = kl_rep.clamp(min=free_nats).mean()

    total = beta_dyn * kl_dyn + beta_rep * kl_rep
    # For logging report the raw (unbalanced) KL magnitude.
    with torch.no_grad():
        kl_raw = kl_categorical(post_l, prior_l).mean()
    return total, kl_raw


class ReturnNorm:
    """
    Official DreamerV3 percentile return normalisation (paper Eq. 7-8):

        S  = EMA( Per(R, 95) - Per(R, 5) )
        R' = R / max(1, S)

    Two properties that matter here:
      * No mean subtraction. Subtracting the mean flips the sign of below-
        average returns, which under sparse reward turns "slightly good"
        into "punished".
      * The max(1, S) floor. Under sparse reward the spread S is tiny, and
        dividing by a tiny number amplifies noise into huge gradients. The
        floor leaves small returns untouched instead of amplifying them.
    """

    def __init__(self, decay=0.99, limit=1.0, lo=0.05, hi=0.95):
        self.decay = decay
        self.limit = limit
        self.lo = lo
        self.hi = hi
        self.ema = None

    @torch.no_grad()
    def scale(self, returns):
        flat = returns.detach().flatten().float()
        lo = torch.quantile(flat, self.lo)
        hi = torch.quantile(flat, self.hi)
        S = (hi - lo).item()
        if not math.isfinite(S):
            S = 0.0
        self.ema = S if self.ema is None else self.decay * self.ema + (1 - self.decay) * S
        return max(self.limit, self.ema)


# ══════════════════════════════════════════════════════════════════════════════
# Normalisation
# ══════════════════════════════════════════════════════════════════════════════

class RMSNorm2d(nn.Module):
    """RMSNorm over channels for conv feature maps (paper uses RMSNorm)."""

    def __init__(self, ch, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, ch, 1, 1))

    def forward(self, x):
        rms = torch.sqrt(x.float().pow(2).mean(1, keepdim=True) + self.eps)
        return (x / rms.to(x.dtype)) * self.g


# ══════════════════════════════════════════════════════════════════════════════
# Encoders / decoders
# ══════════════════════════════════════════════════════════════════════════════

class ImageEncoder(nn.Module):
    """
    Stride-2 conv stack: 64 -> 32 -> 16 -> 8 -> 4, then flatten.
    Uses padded 4x4 kernels so the spatial size halves exactly each layer,
    which makes the decoder a clean mirror.
    """

    def __init__(self, depth=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3,         depth,     4, 2, 1), RMSNorm2d(depth),     nn.SiLU(),
            nn.Conv2d(depth,     depth * 2, 4, 2, 1), RMSNorm2d(depth * 2), nn.SiLU(),
            nn.Conv2d(depth * 2, depth * 4, 4, 2, 1), RMSNorm2d(depth * 4), nn.SiLU(),
            nn.Conv2d(depth * 4, depth * 8, 4, 2, 1), RMSNorm2d(depth * 8), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 3, 64, 64)).shape[1]

    def forward(self, x):
        # uint8 [0,255] -> float [-0.5, 0.5]
        return safe(self.net(x.float() / 255.0 - 0.5))


class VectorEncoder(nn.Module):
    """
    MLP over proprioception. Applies symlog first, as the paper specifies for
    non-image observations: the 30-dim vector mixes joint angles (~pi),
    positions (~0.5 m) and object velocities (potentially >>1), and symlog
    compresses them onto a comparable scale before LayerNorm.
    """

    def __init__(self, in_dim, out_dim=64, units=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, units), nn.LayerNorm(units), nn.SiLU(),
            nn.Linear(units, out_dim), nn.LayerNorm(out_dim), nn.SiLU(),
        )
        self.out_dim = out_dim

    def forward(self, x):
        return safe(self.net(symlog(x)))


class ImageDecoder(nn.Module):
    """Exact mirror of the encoder: 4 -> 8 -> 16 -> 32 -> 64. Output is 64x64."""

    def __init__(self, feat_dim, depth=48):
        super().__init__()
        self.depth = depth
        self.fc = nn.Linear(feat_dim, depth * 8 * 4 * 4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(depth * 8, depth * 4, 4, 2, 1), RMSNorm2d(depth * 4), nn.SiLU(),
            nn.ConvTranspose2d(depth * 4, depth * 2, 4, 2, 1), RMSNorm2d(depth * 2), nn.SiLU(),
            nn.ConvTranspose2d(depth * 2, depth,     4, 2, 1), RMSNorm2d(depth),     nn.SiLU(),
            nn.ConvTranspose2d(depth,     3,         4, 2, 1),
        )
        with torch.no_grad():
            o = self.forward(torch.zeros(1, feat_dim))
            self.out_h, self.out_w = o.shape[2], o.shape[3]

    def forward(self, z):
        x = self.fc(z).view(-1, self.depth * 8, 4, 4)
        return self.net(x)      # linear output; loss is MSE vs [0,1] targets


class VectorDecoder(nn.Module):
    """
    FIX 5 — reconstructs the proprioception/object-state vector from the latent.

    The paper decodes every observation modality. Revision 1 decoded only the
    image, and at 64x64 the can spans a handful of pixels, so the latent had
    almost no gradient pressure to represent object position accurately -- the
    one quantity PickPlaceCan actually depends on. Targets are symlog(vec),
    matching the encoder's input transform and the paper's symlog prediction
    for vector outputs.
    """

    def __init__(self, feat_dim, out_dim, layers=2, units=512):
        super().__init__()
        mods, d = [], feat_dim
        for _ in range(layers):
            mods += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
            d = units
        self.trunk = nn.Sequential(*mods)
        self.out = nn.Linear(d, out_dim)
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, z):
        return self.out(self.trunk(safe(z)))


# ══════════════════════════════════════════════════════════════════════════════
# RSSM
# ══════════════════════════════════════════════════════════════════════════════

class RSSM(nn.Module):
    """
    Recurrent State Space Model.

      h_t = GRU(h_{t-1}, [s_{t-1}, a_{t-1}])      deterministic  (memory)
      prior:      p(s_t | h_t)                     32x32 categorical
      posterior:  q(s_t | h_t, embed_t)            32x32 categorical

    Sampling uses straight-through Gumbel-Softmax so gradients flow into the
    logits while the sampled state stays one-hot. Unimix is applied to the
    logits before sampling and before the KL, matching the paper.

    Note h_t depends only on (s_{t-1}, a_{t-1}) -- the CURRENT observation
    enters the feature vector solely through the posterior sample s_t. If the
    posterior collapses onto the prior, feat = [h, s] contains zero information
    about what the camera is currently seeing. That is why the KL magnitude is
    the single most diagnostic number in the logs.
    """

    def __init__(self, deter, stoch, classes, embed_dim, action_dim,
                 hidden=None, unimix=0.01):
        super().__init__()
        self.deter, self.stoch, self.classes = deter, stoch, classes
        self.lat = stoch * classes
        self.unimix = unimix

        H = hidden or min(deter, 1024)
        self.gru_in = nn.Linear(self.lat + action_dim, deter)
        self.gru_norm = nn.LayerNorm(deter)
        self.gru = nn.GRUCell(deter, deter)

        self.prior = nn.Sequential(
            nn.Linear(deter, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, self.lat))
        self.post = nn.Sequential(
            nn.Linear(deter + embed_dim, H), nn.LayerNorm(H), nn.SiLU(),
            nn.Linear(H, self.lat))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def init_state(self, B, device):
        return (torch.zeros(B, self.deter, device=device),
                torch.zeros(B, self.stoch, self.classes, device=device))

    def feat(self, h, s):
        return torch.cat([h, s.reshape(s.shape[0], -1)], -1)

    def _sample(self, logits):
        logits = unimix_logits(safe(logits).clamp(-15, 15), self.unimix)
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

    def _gru(self, h, s, a):
        x = torch.cat([s.reshape(s.shape[0], -1), a.clamp(-1, 1)], -1)
        x = self.gru_norm(self.gru_in(x))
        return safe(self.gru(safe(x), h)).clamp(-20, 20)

    def obs_step(self, h, s, a, embed):
        """One step WITH an observation. Used for world-model training."""
        h2 = self._gru(h, s, a)
        prior = safe(self.prior(h2)).view(-1, self.stoch, self.classes)
        post = safe(self.post(torch.cat([h2, safe(embed)], -1))
                    ).view(-1, self.stoch, self.classes)
        return h2, self._sample(post), prior, post

    def img_step(self, h, s, a):
        """One step WITHOUT an observation. Used for imagination."""
        h2 = self._gru(h, s, a)
        prior = safe(self.prior(h2)).view(-1, self.stoch, self.classes)
        return h2, self._sample(prior)


# ══════════════════════════════════════════════════════════════════════════════
# Heads
# ══════════════════════════════════════════════════════════════════════════════

class MLPHead(nn.Module):
    """Generic MLP head. zero_out=True zero-inits the final layer (paper)."""

    def __init__(self, in_dim, out_dim, layers=2, units=512, zero_out=False):
        super().__init__()
        mods, d = [], in_dim
        for _ in range(layers):
            mods += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
            d = units
        self.trunk = nn.Sequential(*mods)
        self.out = nn.Linear(d, out_dim)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        if zero_out:
            # Paper: zero-init reward/value outputs so the model predicts
            # exactly 0 at init instead of hallucinating random rewards that
            # the actor would then chase.
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)
        else:
            nn.init.orthogonal_(self.out.weight, gain=0.1)
            nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.trunk(safe(x)))


class TanhNormal:
    """
    Squashed Gaussian for bounded actions in [-1, 1] (OSC_POSE deltas).

    rsample() is reparameterised, so gradients flow from the imagined return
    back through the action into the actor weights. This is the 'dynamics
    backprop' path and is what we use by default.
    """

    def __init__(self, mean, std):
        self.base = torch.distributions.Normal(mean, std)
        self.mean = torch.tanh(mean)
        self._std = std

    def rsample(self):
        return torch.tanh(self.base.rsample())

    def sample(self):
        with torch.no_grad():
            return torch.tanh(self.base.sample())

    def log_prob(self, a):
        a = a.clamp(-0.999999, 0.999999)
        u = torch.atanh(a)
        lp = self.base.log_prob(u) - torch.log1p(-a.pow(2) + 1e-6)
        return lp.sum(-1)

    def entropy(self):
        # Entropy of the PRE-SQUASH Gaussian. NO LONGER USED as the actor's
        # entropy bonus (see FIX 6) because it is monotone increasing in std
        # with no upper turning point: maximising it always pushes std to
        # max_std. Kept for reference and for the analytic ceiling it gives,
        # act_dim * 0.5*log(2*pi*e*max_std^2) = 9.93 at 7 dims, max_std=1.0
        # -- the exact value observed when revision 2 saturated.
        return self.base.entropy().sum(-1)


class Actor(nn.Module):
    def __init__(self, feat_dim, act_dim, layers=3, units=512,
                 min_std=0.1, max_std=1.0):
        super().__init__()
        self.min_std, self.max_std = min_std, max_std
        mods, d = [], feat_dim
        for _ in range(layers):
            mods += [nn.Linear(d, units), nn.LayerNorm(units), nn.SiLU()]
            d = units
        self.trunk = nn.Sequential(*mods)
        self.mean_head = nn.Linear(d, act_dim)
        self.std_head = nn.Linear(d, act_dim)

        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        # Zero-init mean -> policy starts centred (action 0) rather than with
        # an arbitrary bias that must first be unlearned.
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.constant_(self.std_head.bias, 0.0)   # softplus(0)=0.693 -> mid range

    def forward(self, z):
        h = self.trunk(safe(z))
        mean = safe(self.mean_head(h)).clamp(-5, 5)
        std = F.softplus(safe(self.std_head(h))) + self.min_std
        std = std.clamp(self.min_std, self.max_std)
        return TanhNormal(mean, std)

    def act(self, z, deterministic=False):
        d = self.forward(z)
        return d.mean if deterministic else d.sample()


class Critic(nn.Module):
    """Distributional critic: outputs 255 bin logits, read out via TwoHot."""

    def __init__(self, feat_dim, layers=3, units=512, num_bins=255):
        super().__init__()
        self.net = MLPHead(feat_dim, num_bins, layers=layers,
                           units=units, zero_out=True)

    def forward(self, z):
        return self.net(z)


def lambda_returns(rew, val, cont, lam=0.95, gamma=0.997):
    """
    Bootstrapped lambda-returns over the imagined horizon.

      d_t = cont_t * gamma
      R_t = r_t + d_t * ( (1-lam) * v_t + lam * R_{t+1} )

    FIX 2 — gamma is now applied explicitly.

    Revision 1 used cont_t alone as the discount, on the reasoning that the
    continue head folds discount and termination into one term. But with
    horizon=500 terminations are so rare that cont_head collapsed to ~1.0
    (visible as world_model/continue pinned at 0 loss), which meant the agent
    ran at gamma=1.0. Undiscounted bootstrapping lets the value level drift
    freely, which is why returns/mean climbed to ~3 while imagined per-step
    reward was 0.002. The paper's discount horizon is 1/(1-gamma) = 333.
    """
    H = rew.shape[1]
    out = torch.zeros_like(rew)
    nxt = val[:, -1]
    disc = cont * gamma
    for t in reversed(range(H)):
        nxt = rew[:, t] + disc[:, t] * ((1 - lam) * val[:, t] + lam * nxt)
        out[:, t] = nxt
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Environment + instrumentation
# ══════════════════════════════════════════════════════════════════════════════

def make_raw_env(cfg):
    import robosuite as suite
    from robosuite.controllers import load_composite_controller_config
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
        controller_configs=load_composite_controller_config(
            controller=None, robot=cfg["env"]["robot"]),
    )


class InstrumentedEnv:
    """
    Robosuite wrapper that also extracts task-stage metrics.

    DISTANCE
    --------
    Rather than indexing sim.data.site_xpos / body_xpos directly (which
    returned ~17.0 in an earlier revision -- not a metric distance on a
    tabletop), this inverts robosuite's own reach reward, defined as

        r_reach = (1 - tanh(10 * d)) * 0.1
    so
        d = atanh(1 - r_reach / 0.1) / 10

    guaranteed to be exactly the distance robosuite used to compute the
    reward. A direct-sim path is kept as a fallback, sanity-checked against
    a 3 m ceiling.
    """

    REACH_MULT = 0.1
    GRASP_MULT = 0.35
    REACH_THRESHOLD = 0.05      # 5 cm counts as "reached"

    def __init__(self, env, cfg):
        self.env = env
        self.cfg = cfg
        self._low, self._high = env.action_spec
        self.act_dim = self._low.shape[0]
        self._reset_episode_stats()
        self._m = self._blank_metrics()

    # ---- helpers ----------------------------------------------------------

    @staticmethod
    def _blank_metrics():
        return dict(reach=False, grasp=False, lift=False, place=False,
                    dist=float("inf"))

    def _reset_episode_stats(self):
        self.ep_reach = False
        self.ep_grasp = False
        self.ep_lift = False
        self.ep_place = False
        self.ep_min_dist = float("inf")
        self.ep_steps = 0

    # ---- gym-like API -----------------------------------------------------

    def sample_action(self):
        return np.random.uniform(self._low, self._high).astype(np.float32)

    def reset(self):
        self._reset_episode_stats()
        self._m = self._blank_metrics()
        return self._obs(self.env.reset())

    def step(self, action):
        a = np.clip(np.asarray(action, np.float64), self._low, self._high)
        obs, rew, done, info = self.env.step(a)
        self._update(info)
        self.ep_steps += 1
        return self._obs(obs), float(rew), bool(done), info

    def close(self):
        self.env.close()

    def metrics(self):
        return dict(self._m)

    def episode_summary(self):
        if self.ep_place:
            stage = "place"
        elif self.ep_lift:
            stage = "lift"
        elif self.ep_grasp:
            stage = "grasp"
        elif self.ep_reach:
            stage = "reach"
        else:
            stage = "none"
        d = self.ep_min_dist
        return dict(
            reach=self.ep_reach, grasp=self.ep_grasp,
            lift=self.ep_lift, place=self.ep_place,
            min_dist=round(d, 4) if math.isfinite(d) else -1.0,
            steps=self.ep_steps, stage=stage,
        )

    # ---- metric extraction ------------------------------------------------

    def _distance_from_reach(self, r_reach):
        """Invert robosuite's tanh reach reward to recover the true distance."""
        if r_reach <= 0.0:
            return float("inf")
        t = 1.0 - (r_reach / self.REACH_MULT)     # = tanh(10 d)
        t = min(max(t, 0.0), 1.0 - 1e-9)          # keep atanh finite
        return float(np.arctanh(t) / 10.0)

    def _distance_from_sim(self):
        """Fallback: direct sim query, validated for plausibility."""
        try:
            raw = self.env
            obj = raw.objects[0]
            can = np.asarray(raw.sim.data.body_xpos[raw.obj_body_id[obj.name]],
                             dtype=np.float64).reshape(-1)[:3]
            robot = raw.robots[0]
            sid = robot.eef_site_id
            if isinstance(sid, dict):
                sid = sid[list(sid.keys())[0]]
            eef = np.asarray(raw.sim.data.site_xpos[sid],
                             dtype=np.float64).reshape(-1)[:3]
            d = float(np.linalg.norm(eef - can))
            # A tabletop scene is never wider than ~3 m. Anything larger means
            # we indexed the wrong thing, so reject it rather than log garbage.
            return d if 0.0 <= d < 3.0 else float("inf")
        except Exception:
            return float("inf")

    def _update(self, info):
        try:
            raw = self.env
            r_reach, r_grasp, r_lift, _r_hover = raw.staged_rewards()

            dist = self._distance_from_reach(r_reach)
            if not math.isfinite(dist):
                dist = self._distance_from_sim()

            grasp = r_grasp > 0.0
            # r_lift starts exactly at GRASP_MULT the instant a grasp begins and
            # rises toward 0.5 as the can is raised, so require a real margin.
            lift = bool(grasp and r_lift > self.GRASP_MULT + 1e-6)

            place = False
            if hasattr(raw, "objects_in_bins"):
                place = bool(np.sum(raw.objects_in_bins) > 0)
            if not place and isinstance(info, dict):
                place = bool(info.get("success", False)) or place

            reach = math.isfinite(dist) and dist < self.REACH_THRESHOLD

            self._m = dict(reach=reach, grasp=grasp, lift=lift, place=place,
                           dist=round(dist, 4) if math.isfinite(dist) else -1.0)

            self.ep_reach |= reach
            self.ep_grasp |= grasp
            self.ep_lift |= lift
            self.ep_place |= place
            if math.isfinite(dist):
                self.ep_min_dist = min(self.ep_min_dist, dist)
        except Exception:
            # Metrics must never break training.
            pass

    def _obs(self, obs_dict):
        cam = self.cfg["env"]["camera"] + "_image"
        img = np.ascontiguousarray(np.flipud(obs_dict[cam])).transpose(2, 0, 1)
        parts = [np.asarray(obs_dict[k], np.float32).ravel()
                 for k in self.cfg["env"]["obs_keys"] if k in obs_dict]
        vec = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        return {"image": img.copy(), "vector": vec.astype(np.float32)}


# ══════════════════════════════════════════════════════════════════════════════
# Replay buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Stores (image, vector, action, reward, done).

    IMPORTANT alignment convention used throughout this file:
        add(obs_t, action_t, reward_t, done_t)
    where action_t is the action taken *from* obs_t and reward_t is the
    reward received for it. The RSSM is then unrolled with
    obs_step(h, s, a=action_{t-1}, embed=embed_t), i.e. the action that led
    into the current observation. This is the DreamerV3 convention and a
    common source of silent off-by-one bugs.
    """

    def __init__(self, capacity, seq_len):
        self.buf = deque(maxlen=capacity)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.buf)

    def add(self, img, vec, act, rew, done):
        self.buf.append((img.copy(), vec.copy(),
                         np.asarray(act, np.float32),
                         float(rew), bool(done)))

    def ready(self, batch_size):
        return len(self.buf) >= batch_size * self.seq_len + 1

    def sample(self, batch_size, device):
        buf = self.buf
        N = len(buf) - self.seq_len - 1
        idx = np.random.randint(0, max(1, N), batch_size)
        imgs, vecs, acts, rews, dones = [], [], [], [], []
        for i in idx:
            win = [buf[i + t] for t in range(self.seq_len)]
            imgs.append(np.stack([w[0] for w in win]))
            vecs.append(np.stack([w[1] for w in win]))
            acts.append(np.stack([w[2] for w in win]))
            rews.append(np.array([w[3] for w in win], np.float32))
            dones.append(np.array([w[4] for w in win], np.float32))
        t = lambda a: torch.from_numpy(np.stack(a)).to(device, non_blocking=True)
        return t(imgs), t(vecs), t(acts), t(rews), t(dones)

    def reward_stats(self):
        if not self.buf:
            return dict(size=0, mean=0.0, max=0.0, pos_pct=0.0, grasp_pct=0.0)
        r = np.fromiter((x[3] for x in self.buf), np.float32, len(self.buf))
        return dict(
            size=len(self.buf),
            mean=float(r.mean()),
            max=float(r.max()),
            pos_pct=float((r > 1e-4).mean()),
            # grasp contributes 0.35/4 = 0.0875 per step in PickPlaceCan
            grasp_pct=float((r > 0.08).mean()),
        )

    def sanity_scan(self, max_report=5):
        """
        FIX 10 — reject a poisoned buffer at startup.

        After the 84k divergence the acting loop kept writing transitions
        while the networks held NaN, so the tail of the buffer may contain
        non-finite observations or actions. A single such transition will
        re-trigger the blow-up whenever it happens to be sampled, and the
        failure will look spontaneous rather than inherited.

        Returns (n_bad, report_lines).
        """
        if not self.buf:
            return 0, []
        n_bad, lines = 0, []
        for i, (img, vec, act, rew, done) in enumerate(self.buf):
            bad = []
            if not np.isfinite(vec).all():
                bad.append("vector")
            if not np.isfinite(act).all():
                bad.append("action")
            if not math.isfinite(rew):
                bad.append("reward")
            if np.abs(act).max() > 1.0 + 1e-3:
                bad.append(f"|action|={np.abs(act).max():.2f}")
            if bad:
                n_bad += 1
                if len(lines) < max_report:
                    lines.append(f"    idx {i}: {', '.join(bad)}")
        return n_bad, lines

    def drop_tail(self, n):
        """Remove the last n transitions (the post-divergence tail)."""
        for _ in range(min(n, len(self.buf))):
            self.buf.pop()


# ══════════════════════════════════════════════════════════════════════════════
# Training step
# ══════════════════════════════════════════════════════════════════════════════

def train_step(batch, nets, opts, scalers, cfg, device, use_amp,
               twohot, ret_norm, slow_critic):
    """
    One DreamerV3 update:

      1. World model on real replay data
           encoders -> RSSM unroll -> {img decoder, vec decoder, reward, cont}
           loss = beta_pred*(recon_img + recon_vec + reward + cont)
                  + KL_balanced(free bits)
      2. Imagination rollout of length H from posterior states (detached)
      3. Critic  : two-hot regression onto lambda-returns + slow-critic reg
      4. Actor   : maximise percentile-normalised returns (+ entropy bonus)

    Returns a metrics dict, or None if the batch produced non-finite loss.
    """
    (enc_img, enc_vec, rssm, rew_head, cont_head,
     decoder, vec_decoder, actor, critic) = nets
    opt_wm, opt_actor, opt_critic = opts
    sc_wm, sc_actor, sc_critic = scalers

    imgs, vecs, acts, rews, dones = batch
    B, T = imgs.shape[:2]

    wm_cfg = cfg.get("world_model", {})
    ac_cfg = cfg.get("actor_critic", {})

    free_nats = wm_cfg.get("kl_free", 1.0)
    beta_dyn = wm_cfg.get("beta_dyn", 0.5)
    beta_rep = wm_cfg.get("beta_rep", 0.1)
    beta_pred = wm_cfg.get("beta_pred", 1.0)
    unimix = wm_cfg.get("unimix", 0.01)
    wm_clip = wm_cfg.get("grad_clip", 1000.0)          # FIX 3 / rev5: config 50
    vec_scale = wm_cfg.get("vector_loss_scale", 1.0)

    # ── 1. World model ────────────────────────────────────────────────────────
    with torch.amp.autocast("cuda", enabled=use_amp):
        e_img = enc_img(imgs.reshape(B * T, *imgs.shape[2:]))
        e_vec = enc_vec(vecs.reshape(B * T, *vecs.shape[2:]))
        embed = torch.cat([e_img, e_vec], -1).reshape(B, T, -1)

        h, s = rssm.init_state(B, device)
        feats, priors, posts = [], [], []
        zero_a = torch.zeros(B, acts.shape[-1], device=device, dtype=acts.dtype)

        for t in range(T):
            # Action that LED INTO obs_t is acts[t-1]; at t=0 use zeros.
            prev_a = zero_a if t == 0 else acts[:, t - 1]
            if t > 0:
                # Reset latent across episode boundaries inside the window.
                m = (1.0 - dones[:, t - 1]).unsqueeze(-1)
                h = h * m
                s = s * m.unsqueeze(-1)
                prev_a = prev_a * m
            h, s, prior, post = rssm.obs_step(h, s, prev_a, embed[:, t])
            feats.append(rssm.feat(h, s))
            priors.append(prior)
            posts.append(post)

        feats = torch.stack(feats, 1)                 # (B,T,feat)
        priors = torch.stack(priors, 1)
        posts = torch.stack(posts, 1)
        flat = feats.reshape(B * T, -1)

        # ── FIX 1: image reconstruction at PAPER SCALE ──────────────────────
        # Sum over the 3x64x64 pixel dimensions, average over batch*time.
        # F.mse_loss's default reduction='mean' also averages over the 12,288
        # pixel dims, shrinking this term ~12,000x and letting the KL dominate
        # -> posterior collapse. This is the single most important line here.
        tgt_img = imgs.reshape(B * T, 3, 64, 64).float() / 255.0
        rec = decoder(flat)
        if rec.shape[-2:] != tgt_img.shape[-2:]:
            tgt_img = F.interpolate(tgt_img, size=rec.shape[-2:],
                                    mode="bilinear", align_corners=False)
        recon_loss = F.mse_loss(rec, tgt_img, reduction="none") \
                      .sum(dim=(1, 2, 3)).mean()

        # ── FIX 5: vector reconstruction (symlog targets, summed over dims) ──
        if vec_decoder is not None:
            tgt_vec = symlog(vecs.reshape(B * T, -1).float())
            rec_vec = vec_decoder(flat)
            vecrec_loss = F.mse_loss(rec_vec, tgt_vec, reduction="none") \
                           .sum(-1).mean()
        else:
            vecrec_loss = torch.zeros((), device=device)

        # Reward: two-hot cross-entropy (NOT MSE)
        rew_logits = rew_head(flat)
        rew_loss = twohot.loss(rew_logits, rews.reshape(B * T)).mean()

        # Continue: Bernoulli
        cont_logits = cont_head(flat).squeeze(-1)
        cont_loss = F.binary_cross_entropy_with_logits(
            cont_logits, (1.0 - dones).reshape(B * T))

        # KL with balancing + free bits
        kl_loss, kl_raw = kl_balanced(posts, priors, free_nats=free_nats,
                                      beta_dyn=beta_dyn, beta_rep=beta_rep,
                                      unimix=unimix)

        wm_loss = beta_pred * (recon_loss + vec_scale * vecrec_loss
                               + rew_loss + cont_loss) + kl_loss

    # Collapse diagnostics: per-latent entropy in nats, max log(32)=3.466.
    with torch.no_grad():
        post_ent = categorical_entropy(posts.float())
        prior_ent = categorical_entropy(priors.float())

    if not torch.isfinite(wm_loss):
        return None

    opt_wm.zero_grad(set_to_none=True)
    sc_wm.scale(wm_loss).backward()
    sc_wm.unscale_(opt_wm)
    wm_mods = [enc_img, enc_vec, rssm, rew_head, cont_head, decoder]
    if vec_decoder is not None:
        wm_mods.append(vec_decoder)
    wm_params = [p for m in wm_mods for p in m.parameters()]
    wm_gn = nn.utils.clip_grad_norm_(wm_params, wm_clip)

    # ── FIX 8a: refuse the step if the gradient itself is non-finite ─────────
    # clip_grad_norm_ returns inf/nan when any gradient is non-finite, and it
    # does NOT scale the gradients in that case -- it would divide by inf.
    # Stepping here is what wrote NaN into the RSSM parameters at 84k, after
    # which every subsequent batch failed identically. Skipping the step
    # leaves the weights clean and lets the outer loop count the failure.
    if not torch.isfinite(wm_gn):
        opt_wm.zero_grad(set_to_none=True)
        sc_wm.update()
        return None

    sc_wm.step(opt_wm)
    sc_wm.update()

    # ── 2. Imagination ────────────────────────────────────────────────────────
    H = ac_cfg.get("imagination_horizon", 15)
    lam = ac_cfg.get("lambda_", 0.95)
    gamma = ac_cfg.get("discount", 0.997)              # FIX 2 — now used
    ent_coef = ac_cfg.get("actor_entropy", 3e-4)       # FIX 4
    grad_mode = ac_cfg.get("actor_grad", "dynamics")   # 'dynamics' | 'reinforce'
    n_start = ac_cfg.get("imagination_batch", 256)
    a_clip = ac_cfg.get("actor_grad_clip", 100.0)      # rev5: config 5
    c_clip = ac_cfg.get("critic_grad_clip", 100.0)     # rev5: config 5
    slow_reg = ac_cfg.get("critic_slow_reg", 1.0)

    # Detach: the actor must not backprop into the world model's weights.
    starts = feats.reshape(B * T, -1).detach()
    n = min(n_start, starts.shape[0])
    starts = starts[torch.randperm(starts.shape[0], device=device)[:n]]
    Bs = starts.shape[0]
    hi = starts[:, :rssm.deter]
    si = starts[:, rssm.deter:].reshape(Bs, rssm.stoch, rssm.classes)

    im_feats, im_rew, im_cont, im_ent, im_logp = [], [], [], [], []
    with torch.amp.autocast("cuda", enabled=use_amp):
        for _ in range(H):
            f = rssm.feat(hi, si)
            dist = actor(f)
            a = dist.rsample()                       # reparameterised
            # FIX 6: entropy estimated as -log p(a) of the SQUASHED
            # distribution, not the entropy of the pre-squash Gaussian.
            logp = dist.log_prob(a)
            im_ent.append(-logp)
            # REINFORCE needs log p(a) with `a` held fixed (score function);
            # the entropy estimator above needs the pathwise gradient. These
            # are different quantities, so do not share one tensor.
            im_logp.append(dist.log_prob(a.detach())
                           if grad_mode == "reinforce" else logp)
            im_feats.append(f)
            # Rewards/continues stay in the graph so 'dynamics' gradients can
            # flow from the return back through the action.
            im_rew.append(twohot.mean(rew_head(f)))
            im_cont.append(torch.sigmoid(cont_head(f).squeeze(-1)))
            hi, si = rssm.img_step(hi, si, a)

    im_feats = torch.stack(im_feats, 1)               # (Bs,H,feat)
    im_rew = torch.stack(im_rew, 1)                   # (Bs,H)
    im_cont = torch.stack(im_cont, 1)
    im_ent = torch.stack(im_ent, 1)
    im_logp = torch.stack(im_logp, 1)

    # Probability the imagined trajectory is still alive at each step.
    alive = torch.ones_like(im_cont)
    if H > 1:
        alive[:, 1:] = torch.cumprod(im_cont[:, :-1].detach(), dim=1)
    alive_f = alive.reshape(-1)
    alive_sum = alive_f.sum() + 1e-8

    # Values from the SLOW critic (EMA) -> stable bootstrap targets.
    with torch.no_grad():
        vals = twohot.mean(slow_critic(im_feats.reshape(Bs * H, -1))).reshape(Bs, H)

    returns = lambda_returns(im_rew, vals, im_cont, lam=lam, gamma=gamma)

    # ── 3. Critic ─────────────────────────────────────────────────────────────
    tgt = returns.reshape(-1).detach()
    with torch.amp.autocast("cuda", enabled=use_amp):
        c_logits = critic(im_feats.reshape(Bs * H, -1).detach())
        c_loss = (alive_f * twohot.loss(c_logits, tgt)).sum() / alive_sum
        if slow_reg > 0:
            with torch.no_grad():
                slow_val = twohot.mean(
                    slow_critic(im_feats.reshape(Bs * H, -1).detach()))
            c_loss = c_loss + slow_reg * (
                alive_f * twohot.loss(c_logits, slow_val)).sum() / alive_sum

    crit_gn = torch.tensor(0.0)
    if torch.isfinite(c_loss):
        opt_critic.zero_grad(set_to_none=True)
        sc_critic.scale(c_loss).backward()
        sc_critic.unscale_(opt_critic)
        crit_gn = nn.utils.clip_grad_norm_(critic.parameters(), c_clip)
        # FIX 8a (critic)
        if torch.isfinite(crit_gn):
            sc_critic.step(opt_critic)
        else:
            opt_critic.zero_grad(set_to_none=True)
        sc_critic.update()

    # ── 4. Actor ──────────────────────────────────────────────────────────────
    # Percentile scale: divide only, never subtract the mean, and never let the
    # denominator fall below 1. This prevents sparse-reward gradient
    # amplification.
    scale = ret_norm.scale(returns)

    with torch.amp.autocast("cuda", enabled=use_amp):
        if grad_mode == "reinforce":
            adv = ((returns - vals) / scale).reshape(-1).detach()
            objective = im_logp.reshape(-1) * adv
        else:  # 'dynamics' — reparameterised backprop through the world model
            objective = (returns / scale).reshape(-1)

        a_loss = -(alive_f * (objective
                              + ent_coef * im_ent.reshape(-1))).sum() / alive_sum

    actor_gn = torch.tensor(0.0)
    if torch.isfinite(a_loss):
        opt_actor.zero_grad(set_to_none=True)
        sc_actor.scale(a_loss).backward()
        sc_actor.unscale_(opt_actor)
        actor_gn = nn.utils.clip_grad_norm_(actor.parameters(), a_clip)
        # FIX 8a (actor)
        if torch.isfinite(actor_gn):
            sc_actor.step(opt_actor)
        else:
            opt_actor.zero_grad(set_to_none=True)
        sc_actor.update()

    # EMA update of the slow critic
    with torch.no_grad():
        d = ac_cfg.get("critic_ema_decay", 0.98)
        for ps, p in zip(slow_critic.parameters(), critic.parameters()):
            ps.data.mul_(d).add_(p.data, alpha=1 - d)

    f = lambda x: float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
    return dict(
        wm=f(wm_loss), kl=f(kl_raw), recon=f(recon_loss),
        vecrec=f(vecrec_loss), rew=f(rew_loss), cont=f(cont_loss),
        post_ent=f(post_ent), prior_ent=f(prior_ent),
        actor=f(a_loss) if torch.isfinite(a_loss) else float("nan"),
        crit=f(c_loss) if torch.isfinite(c_loss) else float("nan"),
        ret_mean=f(returns.mean()), ret_scale=float(scale),
        imag_rew=f(im_rew.mean()), entropy=f(im_ent.mean()),
        wm_gn=f(wm_gn), actor_gn=f(actor_gn), crit_gn=f(crit_gn),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FIX 8 — snapshot / rewind
# ══════════════════════════════════════════════════════════════════════════════

def snapshot_state(nets_d, opts_d):
    """
    Deep-copy every trainable module AND its optimiser state, on CPU.

    Copying weights alone is the usual way a rewind silently fails: Adam's
    exp_avg / exp_avg_sq buffers are poisoned by a NaN gradient just as the
    weights are, so restoring weights while keeping the moments re-injects
    the corruption on the very next step.

    ~43M params in fp32 plus two Adam moment buffers is roughly 500 MB held
    on CPU. Keep exactly one snapshot; never accumulate a history.
    """
    to_cpu = lambda sd: {k: (v.detach().to("cpu", copy=True)
                             if torch.is_tensor(v) else copy.deepcopy(v))
                         for k, v in sd.items()}
    return {
        "nets": {k: to_cpu(m.state_dict()) for k, m in nets_d.items()},
        "opts": {k: copy.deepcopy(o.state_dict()) for k, o in opts_d.items()},
    }


def restore_state(snap, nets_d, opts_d, device, lr_decay=0.5):
    """Restore a snapshot and decay every learning rate."""
    for k, sd in snap["nets"].items():
        if k in nets_d:
            nets_d[k].load_state_dict({kk: (vv.to(device) if torch.is_tensor(vv)
                                            else vv)
                                       for kk, vv in sd.items()})
    for k, sd in opts_d.items():
        if k in snap["opts"]:
            opts_d[k].load_state_dict(copy.deepcopy(snap["opts"][k]))
    # A rewind WITHOUT a rate reduction just replays the same trajectory into
    # the same cliff. Halving lets the run back off and continue.
    lrs = {}
    for k, o in opts_d.items():
        for g in o.param_groups:
            g["lr"] *= lr_decay
        lrs[k] = o.param_groups[0]["lr"]
    return lrs


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoints + buffer persistence
# ══════════════════════════════════════════════════════════════════════════════

BUFFER_FILE = "replay_buffer.npz"


def find_latest_checkpoint(d):
    p = Path(d) / "latest.pt"
    if p.exists():
        return p
    files = sorted(Path(d).glob("step_*.pt"))
    return files[-1] if files else None


def verify_checkpoint(path):
    """
    FIX 11 — refuse to resume from contaminated weights.

    After the 84k divergence, checkpoints/latest.pt holds post-divergence
    parameters. Resuming from them wastes an entire run before the symptom
    reappears, so scan every tensor before spending compute on it.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bad = []
    for name, sd in ck.get("models", {}).items():
        for k, v in sd.items():
            if torch.is_tensor(v) and not torch.isfinite(v).all():
                bad.append(f"{name}.{k}")
    return bad, ck.get("total_steps", 0)


def save_checkpoint(d, nets_d, opts_d, extra, steps, history, reason):
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    ck = {
        "models": {k: v.state_dict() for k, v in nets_d.items()},
        "optimisers": {k: v.state_dict() for k, v in opts_d.items()},
        "extra": extra,
        "total_steps": steps,
        "ep_ret_history": list(history),
        "saved_at": datetime.now().isoformat(),
        "reason": reason,
    }
    tmp, dest = d / "latest.tmp", d / "latest.pt"
    torch.save(ck, tmp)
    tmp.replace(dest)
    print(f"\n  checkpoint [{reason}] step={steps:,} "
          f"({dest.stat().st_size/1e6:.0f} MB)\n", flush=True)
    return dest


def load_checkpoint(path, nets_d, opts_d, device):
    print(f"\n  resuming from {path}")
    ck = torch.load(path, map_location=device, weights_only=False)
    for k, m in nets_d.items():
        if k in ck["models"]:
            try:
                m.load_state_dict(ck["models"][k])
            except Exception as e:
                print(f"    ! {k}: {e}")
        else:
            print(f"    ! {k}: not in checkpoint (new module) - left at init")
    for k, o in opts_d.items():
        if k in ck.get("optimisers", {}):
            try:
                o.load_state_dict(ck["optimisers"][k])
            except Exception:
                pass
    steps = ck.get("total_steps", 0)
    hist = deque(ck.get("ep_ret_history", []), maxlen=500)
    print(f"  step {steps:,}  (saved {ck.get('saved_at','?')})\n")
    return steps, hist, ck.get("extra", {})


def save_buffer(d, buf):
    if len(buf) == 0:
        return
    d = Path(d)
    t0 = time.time()
    print(f"  saving buffer ({len(buf):,} transitions)...", flush=True)
    imgs = np.stack([x[0] for x in buf.buf])
    vecs = np.stack([x[1] for x in buf.buf])
    acts = np.stack([x[2] for x in buf.buf])
    rews = np.array([x[3] for x in buf.buf], np.float32)
    dones = np.array([x[4] for x in buf.buf], bool)
    stem = d / "replay_buffer_tmp"           # np adds .npz itself
    np.savez_compressed(str(stem), imgs=imgs, vecs=vecs,
                        acts=acts, rews=rews, dones=dones)
    (d / "replay_buffer_tmp.npz").replace(d / BUFFER_FILE)
    sz = (d / BUFFER_FILE).stat().st_size / 1e6
    print(f"  buffer saved ({sz:.0f} MB, {time.time()-t0:.0f}s)")


def load_buffer(d, buf, filename=BUFFER_FILE):
    p = Path(d) / filename
    if not p.exists():
        return False
    t0 = time.time()
    print(f"  loading buffer ({p.stat().st_size/1e6:.0f} MB)...", flush=True)
    z = np.load(str(p), allow_pickle=False)
    imgs, vecs, acts = z["imgs"], z["vecs"], z["acts"]
    rews, dones = z["rews"], z["dones"]
    N = len(imgs)
    start = max(0, N - buf.buf.maxlen)
    for i in range(start, N):
        buf.buf.append((imgs[i], vecs[i], acts[i],
                        float(rews[i]), bool(dones[i])))
    print(f"  buffer loaded: {N-start:,} transitions ({time.time()-t0:.0f}s)")
    return True


# ══════════════════════════════════════════════════════════════════════════════
# Collapse self-check
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def collapse_check(buf, nets, device, n_seq=4):
    """
    Detects the failure mode that killed revision 1.

    If the posterior has collapsed onto the prior, feat = [h, s] carries no
    information from the current frame, and the decoder outputs a near-constant
    image regardless of input. Comparing the decoder's variance ACROSS TIME
    against the target's variance across time makes that visible immediately:
    a healthy model tracks the target's temporal variation, a collapsed one
    emits the mean frame and has near-zero temporal variance.

    NOTE (rev 5): this same ratio is ALSO the earliest divergence warning
    available. It reached 135 during the 84k blow-up -- before the loss
    curves moved -- so main() now trips on it as well as logging it.
    """
    if not buf.ready(n_seq):
        return None
    enc_img, enc_vec, rssm, _, _, decoder, _, _, _ = nets
    imgs, vecs, acts, _, _ = buf.sample(n_seq, device)
    B, T = imgs.shape[:2]
    e = torch.cat([enc_img(imgs.reshape(B * T, *imgs.shape[2:])),
                   enc_vec(vecs.reshape(B * T, -1))], -1).reshape(B, T, -1)
    h, s = rssm.init_state(B, device)
    fs = []
    for t in range(T):
        pa = torch.zeros_like(acts[:, 0]) if t == 0 else acts[:, t - 1]
        h, s, _, _ = rssm.obs_step(h, s, pa, e[:, t])
        fs.append(rssm.feat(h, s))
    rec = decoder(torch.stack(fs, 1).reshape(B * T, -1))
    rec_std = rec.reshape(B, T, -1).float().std(1).mean().item()
    tgt_std = (imgs.float() / 255.0).reshape(B, T, -1).std(1).mean().item()
    ratio = rec_std / (tgt_std + 1e-8)
    verdict = ("COLLAPSED - decoder is emitting a near-constant frame"
               if ratio < 0.15 else
               "weak - latent tracks the scene only loosely" if ratio < 0.4
               else "DIVERGING - decoder variance far above target" if ratio > 5.0
               else "ok")
    print(f"  collapse check: recon_temporal_std={rec_std:.5f} "
          f"target={tgt_std:.5f} ratio={ratio:.3f} -> {verdict}", flush=True)
    return ratio


# ══════════════════════════════════════════════════════════════════════════════
# Logging
# ══════════════════════════════════════════════════════════════════════════════

class StageStats:
    """Rolling per-episode success rates — the real learning signal."""

    def __init__(self, window=20):
        self.w = window
        self.reach = deque(maxlen=window)
        self.grasp = deque(maxlen=window)
        self.lift = deque(maxlen=window)
        self.place = deque(maxlen=window)
        self.dist = deque(maxlen=window)

    def record(self, s):
        self.reach.append(int(s["reach"]))
        self.grasp.append(int(s["grasp"]))
        self.lift.append(int(s["lift"]))
        self.place.append(int(s["place"]))
        if s["min_dist"] >= 0:
            self.dist.append(s["min_dist"])

    def rates(self):
        pct = lambda d: round(float(np.mean(d)) * 100, 1) if d else 0.0
        return dict(
            reach=pct(self.reach), grasp=pct(self.grasp),
            lift=pct(self.lift), place=pct(self.place),
            min_dist=round(float(np.mean(self.dist)), 4) if self.dist else -1.0,
        )


class CSVLog:
    def __init__(self, log_dir, fresh=False):
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        self.lp, self.ep = d / "training_losses.csv", d / "episode_log.csv"
        self._lf = open(self.lp, "w" if fresh or not self.lp.exists() else "a",
                        newline="")
        self._lw = csv.writer(self._lf)
        if self._lf.tell() == 0:
            self._lw.writerow([
                "step", "wm", "kl", "recon", "vecrec", "rew", "cont",
                "post_ent", "prior_ent", "actor", "crit",
                "ret_mean", "ret_scale", "imag_rew", "entropy",
                "wm_gn", "actor_gn", "crit_gn",
                "reach_rate", "grasp_rate", "lift_rate", "place_rate",
                "mean_min_dist", "mean20_return",
                "buf_size", "buf_mean_r", "buf_grasp_pct",
                "nan_skips", "rewinds", "vram_gb", "time"])
            self._lf.flush()
        self._ef = open(self.ep, "w" if fresh or not self.ep.exists() else "a",
                        newline="")
        self._ew = csv.writer(self._ef)
        if self._ef.tell() == 0:
            self._ew.writerow(["episode", "step", "return", "stage",
                               "reach", "grasp", "lift", "place",
                               "min_dist", "ep_steps", "time"])
            self._ef.flush()

    def losses(self, step, L, rates, mean20, bstats, nan, vram, rewinds=0):
        self._lw.writerow([
            step,
            round(L["wm"], 4), round(L["kl"], 4), round(L["recon"], 4),
            round(L["vecrec"], 4), round(L["rew"], 5), round(L["cont"], 5),
            round(L["post_ent"], 4), round(L["prior_ent"], 4),
            round(L["actor"], 5), round(L["crit"], 5),
            round(L["ret_mean"], 5), round(L["ret_scale"], 5),
            round(L["imag_rew"], 6), round(L["entropy"], 4),
            round(L["wm_gn"], 2), round(L["actor_gn"], 2), round(L["crit_gn"], 2),
            rates["reach"], rates["grasp"], rates["lift"], rates["place"],
            rates["min_dist"], round(mean20, 4),
            bstats["size"], round(bstats["mean"], 6),
            round(bstats["grasp_pct"] * 100, 2),
            nan, rewinds, round(vram, 2), datetime.now().strftime("%H:%M:%S")])
        self._lf.flush()

    def episode(self, n, step, ret, s):
        self._ew.writerow([n, step, round(ret, 4), s["stage"],
                           int(s["reach"]), int(s["grasp"]),
                           int(s["lift"]), int(s["place"]),
                           s["min_dist"], s["steps"],
                           datetime.now().strftime("%H:%M:%S")])
        self._ef.flush()

    def close(self):
        self._lf.close()
        self._ef.close()


class WBLog:
    """Weights & Biases: metrics + checkpoint artifacts. Never fatal."""

    STAGES = {"none": 0, "reach": 1, "grasp": 2, "lift": 3, "place": 4}

    def __init__(self, cfg, args, extra_cfg):
        self.on = (WANDB_OK and not args.no_wandb
                   and cfg.get("logging", {}).get("use_wandb", False))
        if not self.on:
            self.run = None
            return
        try:
            self.run = wandb.init(
                project=cfg["logging"].get("wandb_project", "dreamerv3-panda"),
                name=cfg["logging"].get("wandb_run_name")
                or f"dv3_{datetime.now():%Y%m%d_%H%M%S}",
                id=cfg["logging"].get("wandb_run_id"),
                resume="allow", config={**cfg, **extra_cfg}, save_code=True)
            wandb.define_metric("step")
            wandb.define_metric("*", step_metric="step")
            print(f"  wandb: {self.run.url}\n  run id: {self.run.id}")
        except Exception as e:
            print(f"  wandb init failed ({e}) - continuing without it")
            self.on, self.run = False, None

    def losses(self, step, L, rates, mean20, bstats, nan, vram, rewinds=0):
        if not self.on:
            return
        wandb.log({
            "step": step,
            "world_model/loss": L["wm"], "world_model/kl": L["kl"],
            "world_model/reconstruction": L["recon"],
            "world_model/vector_reconstruction": L["vecrec"],
            "world_model/reward": L["rew"], "world_model/continue": L["cont"],
            "world_model/post_entropy": L["post_ent"],
            "world_model/prior_entropy": L["prior_ent"],
            "actor/loss": L["actor"], "actor/entropy": L["entropy"],
            "actor/grad_norm": L["actor_gn"],
            "critic/loss": L["crit"], "critic/grad_norm": L["crit_gn"],
            "returns/mean": L["ret_mean"], "returns/scale": L["ret_scale"],
            "returns/imagined_reward": L["imag_rew"],
            "world_model/grad_norm": L["wm_gn"],
            "policy/reach_rate": rates["reach"], "policy/grasp_rate": rates["grasp"],
            "policy/lift_rate": rates["lift"], "policy/place_rate": rates["place"],
            "policy/min_distance": rates["min_dist"],
            "reward/rolling_return": mean20,
            "buffer/size": bstats["size"], "buffer/mean_reward": bstats["mean"],
            "buffer/grasp_pct": bstats["grasp_pct"] * 100,
            "train/nan_skips": nan, "train/rewinds": rewinds,
            "gpu/vram_gb": vram,
        })

    def scalar(self, step, key, value):
        if self.on:
            wandb.log({"step": step, key: value})

    def episode(self, step, ret, s, n):
        if not self.on:
            return
        wandb.log({
            "step": step, "episode/return": ret, "episode/number": n,
            "episode/stage": self.STAGES.get(s["stage"], 0),
            "episode/min_distance": s["min_dist"],
            "episode/length": s["steps"],
            "episode/reached": int(s["reach"]), "episode/grasped": int(s["grasp"]),
            "episode/lifted": int(s["lift"]), "episode/placed": int(s["place"]),
        })

    def artifact(self, path, steps, reason):
        if not self.on:
            return
        try:
            art = wandb.Artifact(f"dv3-panda-{self.run.id}", type="model",
                                 metadata={"step": steps, "reason": reason})
            art.add_file(str(path), name="latest.pt")
            self.run.log_artifact(art, aliases=["latest", f"step-{steps}"])
        except Exception as e:
            print(f"  wandb artifact skipped: {e}")

    def finish(self):
        if self.on:
            try:
                wandb.finish()
            except Exception:
                pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = get_device(args.device)
    use_amp = cfg["training"].get("mixed_precision", True) and device.type == "cuda"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"\nDevice   : {device_info(device)}")
    print(f"AMP fp16 : {use_amp}")
    if use_amp:
        print("  WARNING (rev 5): mixed_precision is ON. fp16 overflow is the")
        print("  prime suspect for the 84k divergence -- reconstruction reached")
        print("  ~2000 and fp16 maxes out at 65504. Set mixed_precision: false")
        print("  unless you are deliberately reproducing that failure.")

    # ── revision-5 stability settings ─────────────────────────────────────────
    stab = cfg.get("stability", {})
    NAN_PATIENCE   = stab.get("nan_patience", 20)
    SNAPSHOT_EVERY = stab.get("snapshot_every", 500)
    LR_DECAY       = stab.get("lr_decay_on_rewind", 0.5)
    MAX_REWINDS    = stab.get("max_rewinds", 4)
    ABORT_RECON    = stab.get("abort_on_recon", 100.0)
    ABORT_RATIO    = stab.get("abort_on_std_ratio", 5.0)
    DROP_BAD_TAIL  = stab.get("drop_bad_buffer_tail", 2000)

    ckpt_dir = Path(cfg["logging"]["checkpoint_dir"])
    log_dir = Path(cfg["logging"]["log_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── environment ───────────────────────────────────────────────────────────
    print("Building environment...")
    env = InstrumentedEnv(make_raw_env(cfg), cfg)
    act_dim = env.act_dim
    obs = env.reset()
    vec_dim = obs["vector"].shape[0]
    print(f"act_dim={act_dim}  vec_dim={vec_dim}")

    # ── networks ──────────────────────────────────────────────────────────────
    wm_cfg, ac_cfg = cfg["world_model"], cfg["actor_critic"]
    depth = wm_cfg["encoder_cnn_depth"]
    deter = wm_cfg["rssm_deter"]
    stoch = wm_cfg["rssm_stoch"]
    classes = wm_cfg["rssm_classes"]
    num_bins = wm_cfg.get("num_bins", 255)
    feat_dim = deter + stoch * classes

    enc_img = ImageEncoder(depth).to(device)
    enc_vec = VectorEncoder(vec_dim, 64,
                            wm_cfg.get("encoder_mlp_units", 256)).to(device)
    embed_dim = enc_img.out_dim + enc_vec.out_dim

    rssm = RSSM(deter, stoch, classes, embed_dim, act_dim,
                unimix=wm_cfg.get("unimix", 0.01)).to(device)
    decoder = ImageDecoder(feat_dim, depth).to(device)

    vec_decoder = None
    if wm_cfg.get("decode_vector", True):
        vec_decoder = VectorDecoder(
            feat_dim, vec_dim,
            layers=wm_cfg.get("vector_decoder_layers", 2),
            units=wm_cfg.get("vector_decoder_units", 512)).to(device)

    rew_head = MLPHead(feat_dim, num_bins,
                       layers=wm_cfg.get("reward_layers", 2),
                       units=wm_cfg.get("reward_units", 512),
                       zero_out=True).to(device)
    cont_head = MLPHead(feat_dim, 1,
                        layers=wm_cfg.get("continue_layers", 2),
                        units=wm_cfg.get("continue_units", 512)).to(device)
    actor = Actor(feat_dim, act_dim,
                  layers=ac_cfg.get("actor_layers", 3),
                  units=ac_cfg.get("actor_units", 512),
                  min_std=ac_cfg.get("actor_min_std", 0.1),
                  max_std=ac_cfg.get("actor_max_std", 1.0)).to(device)
    critic = Critic(feat_dim,
                    layers=ac_cfg.get("critic_layers", 3),
                    units=ac_cfg.get("critic_units", 512),
                    num_bins=num_bins).to(device)
    slow_critic = copy.deepcopy(critic).to(device)
    for p in slow_critic.parameters():
        p.requires_grad_(False)

    twohot = TwoHot(num_bins=num_bins, device=device)
    ret_norm = ReturnNorm(decay=ac_cfg.get("return_ema_decay", 0.99),
                          limit=ac_cfg.get("return_limit", 1.0))

    all_mods = [enc_img, enc_vec, rssm, decoder, rew_head, cont_head,
                actor, critic]
    if vec_decoder is not None:
        all_mods.append(vec_decoder)
    n_params = sum(p.numel() for m in all_mods for p in m.parameters())
    print(f"Parameters : {n_params:,}")
    print(f"Decoder out: {decoder.out_h}x{decoder.out_w}")
    print(f"Vec decoder: {'on' if vec_decoder is not None else 'off'}"
          f"  (loss scale {wm_cfg.get('vector_loss_scale', 1.0)})")
    print(f"Actor grad : {ac_cfg.get('actor_grad', 'dynamics')}")
    print(f"Actor std  : [{ac_cfg.get('actor_min_std', 0.1)}, "
          f"{ac_cfg.get('actor_max_std', 1.0)}]")
    print(f"Grad clips : wm={wm_cfg.get('grad_clip', 1000.0)} "
          f"actor={ac_cfg.get('actor_grad_clip', 100.0)} "
          f"critic={ac_cfg.get('critic_grad_clip', 100.0)}")
    print(f"Discount   : {ac_cfg.get('discount', 0.997)}  "
          f"(horizon {1/(1-ac_cfg.get('discount', 0.997)):.0f})")
    print(f"Rewind     : patience={NAN_PATIENCE} snapshot_every={SNAPSHOT_EVERY} "
          f"lr_decay={LR_DECAY} max={MAX_REWINDS}")

    # ── optimisers ────────────────────────────────────────────────────────────
    wm_mods = [enc_img, enc_vec, rssm, decoder, rew_head, cont_head]
    if vec_decoder is not None:
        wm_mods.append(vec_decoder)
    wm_params = [p for m in wm_mods for p in m.parameters()]
    opt_wm = torch.optim.Adam(wm_params, lr=wm_cfg["model_lr"],
                              eps=wm_cfg.get("model_eps", 1e-8))
    opt_actor = torch.optim.Adam(actor.parameters(), lr=ac_cfg["actor_lr"], eps=1e-5)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=ac_cfg["critic_lr"], eps=1e-5)

    sc_wm = torch.amp.GradScaler("cuda", enabled=use_amp)
    sc_actor = torch.amp.GradScaler("cuda", enabled=use_amp)
    sc_critic = torch.amp.GradScaler("cuda", enabled=use_amp)

    nets = (enc_img, enc_vec, rssm, rew_head, cont_head,
            decoder, vec_decoder, actor, critic)
    opts = (opt_wm, opt_actor, opt_critic)
    scalers = (sc_wm, sc_actor, sc_critic)

    nets_d = dict(enc_img=enc_img, enc_vec=enc_vec, rssm=rssm, decoder=decoder,
                  rew_head=rew_head, cont_head=cont_head,
                  actor=actor, critic=critic, slow_critic=slow_critic)
    if vec_decoder is not None:
        nets_d["vec_decoder"] = vec_decoder
    opts_d = dict(opt_wm=opt_wm, opt_actor=opt_actor, opt_critic=opt_critic)

    # ── logging ───────────────────────────────────────────────────────────────
    csv_log = CSVLog(log_dir, fresh=args.fresh)
    wb = WBLog(cfg, args, dict(total_params=n_params,
                               vec_dim=vec_dim, act_dim=act_dim,
                               revision=5))

    # ── replay + resume ───────────────────────────────────────────────────────
    buf = ReplayBuffer(cfg["replay"]["capacity"], cfg["replay"]["batch_length"])
    PREFILL = cfg["training"]["prefill_steps"]
    BATCH = cfg["replay"]["batch_size"]

    steps = 0
    hist = deque(maxlen=500)
    if not args.fresh:
        # FIX 11 — explicit --checkpoint takes priority over latest.pt.
        if args.checkpoint:
            latest = Path(args.checkpoint)
            if not latest.exists():
                raise FileNotFoundError(f"--checkpoint not found: {latest}")
        else:
            latest = find_latest_checkpoint(ckpt_dir)
            if latest:
                print("\n  NOTE: no --checkpoint given, using latest.pt.")
                print("  If you are recovering from the 84k divergence this is")
                print("  the WRONG file -- latest.pt was overwritten with")
                print("  post-divergence weights. Pass the 80k artifact")
                print("  explicitly with --checkpoint.\n")

        if latest:
            bad, ck_steps = verify_checkpoint(latest)
            if bad:
                print(f"\n  ABORT: {len(bad)} non-finite tensors in {latest}")
                for b in bad[:10]:
                    print(f"    {b}")
                print("  This checkpoint is contaminated. Fall back to an")
                print("  earlier one (e.g. the 55k checkpoint).\n")
                raise RuntimeError("contaminated checkpoint")
            print(f"  checkpoint verified clean (step {ck_steps:,})")

            steps, hist, extra = load_checkpoint(latest, nets_d, opts_d, device)

            # FIX 12 — the old FIX 7 is now opt-in. Resuming rev4 -> rev5
            # changes NO actor hyperparameter, so discarding the actor's Adam
            # moments would throw away 25k steps of adaptation for nothing.
            # Use --reset_actor_opt only when max_std / min_std / entropy
            # scale changed relative to the checkpoint.
            if args.reset_actor_opt:
                opt_actor = torch.optim.Adam(actor.parameters(),
                                             lr=ac_cfg["actor_lr"], eps=1e-5)
                opts_d["opt_actor"] = opt_actor
                opts = (opt_wm, opt_actor, opt_critic)
                print("  actor optimiser state reset (FIX 7, requested)")
            else:
                print("  actor optimiser state KEPT (pass --reset_actor_opt "
                      "only if an actor hyperparameter changed)")

            if extra.get("ret_ema") is not None:
                ret_norm.ema = extra["ret_ema"]
            if not args.no_save_buffer and not load_buffer(ckpt_dir, buf):
                n = min(PREFILL, max(1000, steps // 10))
                print(f"  no saved buffer - collecting {n:,} random steps")
                o = env.reset()
                for _ in range(n):
                    a = env.sample_action()
                    o2, r, d, _ = env.step(a)
                    buf.add(o["image"], o["vector"], a, r, d)
                    o = env.reset() if d else o2
        else:
            print("  no checkpoint - starting fresh\n")
    else:
        print("  --fresh: ignoring checkpoint and buffer\n")

    # ── FIX 10: buffer sanity scan ────────────────────────────────────────────
    if len(buf) > 0:
        t0s = time.time()
        n_bad, lines = buf.sanity_scan()
        print(f"  buffer scan: {len(buf):,} transitions, {n_bad} bad "
              f"({time.time()-t0s:.0f}s)")
        for ln in lines:
            print(ln)
        if n_bad:
            if DROP_BAD_TAIL > 0:
                print(f"  dropping last {DROP_BAD_TAIL:,} transitions "
                      f"(post-divergence tail) and rescanning")
                buf.drop_tail(DROP_BAD_TAIL)
                n_bad, _ = buf.sanity_scan()
                print(f"  after drop: {len(buf):,} transitions, {n_bad} bad")
            if n_bad:
                raise RuntimeError(
                    "replay buffer still contains non-finite data -- resume "
                    "from an earlier buffer or raise stability.drop_bad_buffer_tail")

    obs = env.reset()

    # top up prefill if needed
    if len(buf) < PREFILL:
        need = PREFILL - len(buf)
        print(f"Prefilling {need:,} random steps ({len(buf):,} already stored)...")
        for _ in range(need):
            a = env.sample_action()
            o2, r, d, _ = env.step(a)
            buf.add(obs["image"], obs["vector"], a, r, d)
            obs = env.reset() if d else o2
        obs = env.reset()
        print(f"Buffer ready: {len(buf):,}\n")

    start_ratio = collapse_check(buf, nets, device)
    if start_ratio is not None and start_ratio > ABORT_RATIO:
        raise RuntimeError(
            f"recon_std_ratio={start_ratio:.1f} at startup (limit {ABORT_RATIO}). "
            "These weights are already diverging -- use an earlier checkpoint.")

    # ── training state ────────────────────────────────────────────────────────
    t0 = time.time()
    h_s, s_s = rssm.init_state(1, device)
    prev_a = torch.zeros(1, act_dim, device=device)
    ep_ret, nan_skips, L = 0.0, 0, None
    consec_nan, rewinds, grad_step = 0, 0, 0     # rev 5
    good_state = None                             # rev 5
    stop = False
    abort_reason = None
    stages = StageStats(window=cfg["training"].get("stat_window", 20))
    episode_n = len(hist)
    max_sec = args.max_minutes * 60 if args.max_minutes > 0 else float("inf")

    def _sigint(sig, frame):
        nonlocal stop
        print("\n\n  interrupt - saving before exit...")
        stop = True
    signal.signal(signal.SIGINT, _sigint)

    print(f"Training from step {steps:,}\n")

    TOTAL = cfg["training"]["total_steps"]
    RATIO = cfg["training"]["train_ratio"]
    LOG_EVERY = cfg["training"]["log_every"]
    CKPT_EVERY = cfg["training"]["checkpoint_every"]
    CHECK_EVERY = cfg["training"].get("collapse_check_every", 2000)

    # ── main loop ─────────────────────────────────────────────────────────────
    while steps < TOTAL and not stop and (time.time() - t0) < max_sec:

        # act
        with torch.no_grad():
            it = torch.from_numpy(obs["image"]).unsqueeze(0).to(device)
            vt = torch.from_numpy(obs["vector"]).unsqueeze(0).to(device)
            emb = torch.cat([enc_img(it), enc_vec(vt)], -1)
            h_s, s_s, _, _ = rssm.obs_step(h_s, s_s, prev_a, emb)
            if steps < PREFILL:
                a = env.sample_action()
            else:
                a = actor.act(rssm.feat(h_s, s_s)).squeeze(0).cpu().numpy()
                a = np.clip(a, -1, 1).astype(np.float32)
            # rev 5: never write a non-finite action into the buffer, however
            # the networks are behaving. This is what poisons a restart.
            if not np.isfinite(a).all():
                a = np.zeros_like(a)

        nxt, r, done, _ = env.step(a)
        # add(obs_t, a_t, r_t, done_t) — see ReplayBuffer docstring
        if np.isfinite(obs["vector"]).all() and math.isfinite(r):
            buf.add(obs["image"], obs["vector"], a, r, done)
        prev_a = torch.from_numpy(np.asarray(a, np.float32)).unsqueeze(0).to(device)
        ep_ret += r
        obs = nxt
        steps += 1

        if done:
            s = env.episode_summary()
            stages.record(s)
            hist.append(ep_ret)
            episode_n += 1
            m20 = float(np.mean(list(hist)[-20:]))
            rt = stages.rates()
            print(f"[{steps:>9,}] ret={ep_ret:7.3f} mean20={m20:6.3f} "
                  f"stage={s['stage']:<5} d={s['min_dist']:.3f} "
                  f"| R{rt['reach']:.0f} G{rt['grasp']:.0f} "
                  f"L{rt['lift']:.0f} P{rt['place']:.0f} "
                  f"| {fmt_time(time.time()-t0)}", flush=True)
            csv_log.episode(episode_n, steps, ep_ret, s)
            wb.episode(steps, ep_ret, s, episode_n)
            obs = env.reset()
            h_s, s_s = rssm.init_state(1, device)
            prev_a = torch.zeros(1, act_dim, device=device)
            ep_ret = 0.0

        # learn
        if steps >= PREFILL and buf.ready(BATCH):
            for _ in range(RATIO):
                out = train_step(buf.sample(BATCH, device), nets, opts, scalers,
                                 cfg, device, use_amp, twohot, ret_norm, slow_critic)
                grad_step += 1

                if out is None:
                    # ── FIX 8: detect AND recover ────────────────────────────
                    nan_skips += 1
                    consec_nan += 1
                    if consec_nan >= NAN_PATIENCE:
                        if good_state is None:
                            abort_reason = (
                                f"{consec_nan} consecutive non-finite batches "
                                f"and no clean snapshot yet")
                            stop = True
                            break
                        rewinds += 1
                        lrs = restore_state(good_state, nets_d, opts_d,
                                            device, lr_decay=LR_DECAY)
                        print(f"\n  [rev5] REWIND {rewinds} at step {steps:,} "
                              f"(grad_step {grad_step:,}) after {consec_nan} "
                              f"failed batches")
                        print(f"         LRs -> " +
                              "  ".join(f"{k}={v:.2e}" for k, v in lrs.items()),
                              flush=True)
                        wb.scalar(steps, "train/rewinds", rewinds)
                        wb.scalar(steps, "train/lr_wm",
                                  opts_d["opt_wm"].param_groups[0]["lr"])
                        consec_nan = 0
                        if rewinds > MAX_REWINDS:
                            abort_reason = (
                                f"{rewinds} rewinds -- configuration is "
                                f"unsalvageable at this operating point")
                            stop = True
                            break
                else:
                    consec_nan = 0
                    L = out
                    # ── FIX 9: tripwire, before the corruption lands ─────────
                    if L["recon"] > ABORT_RECON:
                        abort_reason = (f"reconstruction={L['recon']:.1f} "
                                        f"exceeded limit {ABORT_RECON}")
                        stop = True
                        break
                    # snapshot only from a healthy state
                    if grad_step % SNAPSHOT_EVERY == 0:
                        good_state = snapshot_state(nets_d, opts_d)

            if stop:
                break

            if steps % LOG_EVERY == 0 and L is not None:
                vram = (torch.cuda.memory_allocated(0) / 1e9
                        if device.type == "cuda" else 0.0)
                bs = buf.reward_stats()
                rt = stages.rates()
                m20 = float(np.mean(list(hist)[-20:])) if hist else 0.0
                print(f"   wm={L['wm']:.2f} kl={L['kl']:.3f} "
                      f"rec={L['recon']:.2f} vrec={L['vecrec']:.2f} "
                      f"rew={L['rew']:.4f} pH={L['post_ent']:.2f} "
                      f"act={L['actor']:+.4f} crit={L['crit']:.4f} "
                      f"| S={L['ret_scale']:.3f} wgn={L['wm_gn']:.1f} "
                      f"agn={L['actor_gn']:.1f} "
                      f"| bufG={bs['grasp_pct']*100:.1f}% nan={nan_skips} "
                      f"rw={rewinds} vram={vram:.1f}G", flush=True)
                csv_log.losses(steps, L, rt, m20, bs, nan_skips, vram, rewinds)
                wb.losses(steps, L, rt, m20, bs, nan_skips, vram, rewinds)

            if CHECK_EVERY > 0 and steps % CHECK_EVERY == 0:
                ratio = collapse_check(buf, nets, device)
                if ratio is not None:
                    wb.scalar(steps, "world_model/recon_std_ratio", ratio)
                    # FIX 9: recon_std_ratio moves BEFORE the loss does. It
                    # reached 135 during the 84k event.
                    if ratio > ABORT_RATIO:
                        abort_reason = (f"recon_std_ratio={ratio:.1f} exceeded "
                                        f"limit {ABORT_RATIO}")
                        stop = True

        # checkpoint
        if steps % CKPT_EVERY == 0 and steps > 0:
            p = save_checkpoint(ckpt_dir, nets_d, opts_d,
                                dict(ret_ema=ret_norm.ema), steps, hist, "periodic")
            if not args.no_save_buffer:
                save_buffer(ckpt_dir, buf)
            wb.artifact(p, steps, "periodic")

    # ── shutdown ──────────────────────────────────────────────────────────────
    if abort_reason:
        print(f"\n  [rev5] TRIPWIRE at step {steps:,}: {abort_reason}")
        print("  Saving now, while the weights are still usable.\n")
        reason = "tripwire"
    else:
        reason = ("interrupted" if stop else
                  "time_limit" if (time.time() - t0) >= max_sec else "completed")

    p = save_checkpoint(ckpt_dir, nets_d, opts_d,
                        dict(ret_ema=ret_norm.ema), steps, hist, reason)
    if not args.no_save_buffer:
        save_buffer(ckpt_dir, buf)
    wb.artifact(p, steps, reason)
    wb.finish()
    csv_log.close()
    env.close()
    print(f"\nDone. steps={steps:,} time={fmt_time(time.time()-t0)} ({reason})")
    print(f"      nan_skips={nan_skips:,}  rewinds={rewinds}")


if __name__ == "__main__":
    main()