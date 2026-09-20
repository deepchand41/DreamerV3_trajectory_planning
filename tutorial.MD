# Robosuite from Zero to a Working RL Agent

### A tutorial for people who have never simulated a robot before

---

## Part 0 — What robosuite actually is

Before installing anything, get the mental model right. Most beginner confusion comes from not knowing which layer a problem lives in.

Robosuite is **three layers stacked on top of each other**:

```
┌──────────────────────────────────────────────────┐
│  YOUR CODE          agent, training loop         │
├──────────────────────────────────────────────────┤
│  ROBOSUITE          tasks, robots, controllers,  │
│                     rewards, cameras, objects    │
├──────────────────────────────────────────────────┤
│  MUJOCO             rigid-body physics, contacts,│
│                     integration, rendering       │
└──────────────────────────────────────────────────┘
```

MuJoCo knows about masses, joints, friction and collisions. It does not know what "picking up a can" means. Robosuite is the layer that says: *here is a Panda arm, here is a table, here is a can, here is what counts as success, here is a camera pointed at it, and here is how to turn a 7-number action into joint torques.*

**Why this matters:** when something breaks, ask which layer. Rendering errors and `libGL` failures are MuJoCo. "Object not found", wrong action dimensions and weird rewards are robosuite. NaN losses are your code.

### The one loop that runs everything

Every simulation, every RL algorithm, every paper — all of it reduces to this:

```
obs = env.reset()
loop:
    action = policy(obs)
    obs, reward, done, info = env.step(action)
    if done: obs = env.reset()
```

That's it. Everything else in this tutorial is detail about what `obs` contains, what `action` means, where `reward` comes from, and how `policy` learns.

---

## Part 1 — Installation

### The environment

Use conda. Python 3.10 or 3.11 — **not 3.12**, which has caused MuJoCo wheel problems.

```bash
conda create -n robo python=3.11 -y
conda activate robo

pip install torch numpy
pip install mujoco
pip install robosuite
```

Check it worked:

```bash
python -c "import robosuite; print(robosuite.__version__)"
python -c "import mujoco; print(mujoco.__version__)"
```

### Headless rendering (servers, Colab, Docker)

If you have no monitor, MuJoCo needs to be told to render off-screen. Set this **before importing anything**:

```python
import os
os.environ["MUJOCO_GL"] = "egl"        # or "osmesa" if EGL is unavailable
os.environ["PYOPENGL_PLATFORM"] = "egl"

import robosuite   # only now
```

Getting the order wrong is the single most common installation failure. The GL backend is chosen at import time; setting the variable afterwards does nothing.

On Colab you also need:

```bash
apt-get install -y libegl1-mesa libgl1-mesa-glx libosmesa6
```

### Apple Silicon

Native ARM works but EGL does not. Use `MUJOCO_GL=glfw` with a display, or accept CPU-only rendering. Never run robosuite under x86 emulation — it is roughly 10× slower.

---

## Part 2 — Your first script

```python
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

env = suite.make(
    env_name="Lift",              # the simplest task: lift a cube
    robots="Panda",
    has_renderer=False,           # no interactive window
    has_offscreen_renderer=False, # no camera images (faster)
    use_camera_obs=False,         # observations are numbers only
    reward_shaping=True,          # dense reward (explained in Part 6)
    control_freq=20,              # 20 agent decisions per simulated second
    horizon=200,                  # episode ends after 200 steps
    controller_configs=load_composite_controller_config(
        controller=None, robot="Panda"),
)

obs = env.reset()

print("Observation keys:")
for k, v in obs.items():
    print(f"  {k:30s} {np.array(v).shape}")

low, high = env.action_spec
print(f"\nAction dimension: {low.shape[0]}")
print(f"Action range: [{low[0]:.1f}, {high[0]:.1f}]")

total = 0.0
for t in range(200):
    action = np.random.uniform(low, high)
    obs, reward, done, info = env.step(action)
    total += reward
    if done:
        break

print(f"\nRandom policy return over one episode: {total:.3f}")
env.close()
```

Run it. You should see a list of observation keys, an action dimension of 7, and a small positive return.

**What just happened.** You built a MuJoCo scene containing a Panda arm and a cube, reset it to a random initial configuration, and drove it with 200 random actions. The arm flailed. It scored a little reward anyway, because the reward function gives partial credit for being near the cube, and random flailing sometimes passes near it.

---

## Part 3 — The five things that define an environment

Every `suite.make()` call is answering five questions. Understanding these five is understanding robosuite.

### 1. The task — *what are we trying to do?*

```python
env_name="Lift"          # lift a cube off the table
env_name="PickPlaceCan"  # pick up a can, put it in the right bin
env_name="NutAssembly"   # place a nut onto a peg
env_name="Door"          # open a door
env_name="Stack"         # stack one block on another
env_name="Wipe"          # wipe markings off a table
```

Difficulty ordering, roughly: `Lift` < `Door` < `Stack` < `PickPlaceCan` < `NutAssembly`.

**Start with `Lift`.** Always. It is the "hello world" of manipulation: one object, one clear goal, reachable in about 50 steps. If your code can't solve `Lift`, it will not solve anything else, and `Lift` tells you so in an hour instead of a week.

### 2. The robot — *what body are we controlling?*

```python
robots="Panda"     # 7-DoF Franka Emika Panda. The default. Start here.
robots="Sawyer"    # 7-DoF Rethink Sawyer
robots="UR5e"      # 6-DoF Universal Robots
robots="Kinova3"   # 7-DoF
robots=["Panda", "Panda"]   # two-arm tasks
```

Robosuite 1.5 <cite index="41-1">added support for diverse robot embodiments including humanoids, custom robot composition, and composite controllers.</cite> Ignore all of that for now. Use `Panda`.

### 3. The controller — *what does an action number mean?*

This gets its own section (Part 4) because it is where most beginners go wrong.

### 4. The observations — *what does the agent see?*

Also its own section (Part 5).

### 5. The reward — *what counts as good?*

Also its own section (Part 6).

### Timing arguments you should understand

```python
control_freq=20     # agent decisions per second of simulated time
horizon=200         # max steps before the episode is cut off
```

MuJoCo integrates physics at ~500 Hz internally. At `control_freq=20`, each `env.step()` advances physics by 25 internal ticks and holds your action constant throughout. So one agent step = 0.05 simulated seconds.

This means `horizon=200` is **10 seconds of robot time**, not 200 seconds. For `Lift` that's plenty. For `PickPlaceCan` — reach, grasp, lift, carry, release — you want 500.

**A useful intuition:** raising `control_freq` gives finer control but makes the credit-assignment problem harder, because any given outcome is now the result of more actions. 20 Hz is a good default and is what almost all published robosuite results use.

---

## Part 4 — Controllers: the beginner trap

An action is a vector of numbers. **What those numbers mean is entirely determined by the controller.** Same array, completely different robot behaviour.

### The two you'll actually use

**`JOINT_VELOCITY`** — action is a target angular velocity for each joint.

```
action = [ j1, j2, j3, j4, j5, j6, j7, gripper ]   # 8 numbers for Panda
```

Direct and predictable. But to move the gripper 5 cm to the left, the agent has to work out the seven coordinated joint velocities that produce that motion. It has to learn inverse kinematics from scratch, from reward alone.

**`OSC_POSE`** (Operational Space Control) — action is a desired *end-effector* motion.

```
action = [ dx, dy, dz, drx, dry, drz, gripper ]    # 7 numbers for Panda
          └── position ──┘ └── rotation ──┘
```

The controller solves the inverse kinematics for you. "Move 5 cm left" is now literally `[0, 0.05, 0, 0, 0, 0, ...]` scaled into range.

### Which to choose

**Use `OSC_POSE` for manipulation.** It is not a small convenience — it removes an entire learning problem. Published results on robosuite manipulation overwhelmingly use OSC.

`JOINT_VELOCITY` is for when you specifically want to study low-level control or whole-body motion.

### Loading a controller

In robosuite 1.5 the API is `load_composite_controller_config`, which <cite index="33-1">loads the desired composite controller and returns the configuration as a dict.</cite>

```python
from robosuite.controllers import load_composite_controller_config

# BASIC composite controller: arms use OSC_POSE by default
cfg = load_composite_controller_config(controller=None, robot="Panda")

env = suite.make(..., controller_configs=cfg)
```

<cite index="35-1">The "Basic" composite controller consists of individual part controllers operating independently to control various parts of the robot — arms, torso, head, base, legs — with each part assigned a controller type such as OSC_POSE or JOINT_POSITION.</cite> For a fixed-base Panda there's only one part that matters, the arm, and it defaults to OSC.

> **Version warning.** Robosuite 1.4 used `load_controller_config(default_controller="OSC_POSE")`. That function is gone in 1.5. If you're following an older tutorial and get `ImportError`, this is why. Check your version first: `robosuite.__version__`.

### Always read the action space from the environment

Never hardcode the action dimension. It changes with robot, controller and gripper:

```python
low, high = env.action_spec
act_dim = low.shape[0]
print(f"actions are {act_dim}-dimensional in range [{low[0]}, {high[0]}]")
```

Robosuite normalises actions to `[-1, 1]`. Your policy should output in that range — a `tanh` on the network output is the standard way — and you should still clip before stepping, because floating-point noise at the boundary can trip assertions.

### The gripper dimension is not like the others

The last action element controls the gripper: **negative opens, positive closes**. It's effectively binary even though it's continuous. A policy that outputs `-0.01` is barely opening; one that outputs `-1.0` opens fully. Early in training a policy that hovers near zero will barely actuate the gripper at all — a common reason agents seem to "almost" grasp for a long time.

---

## Part 5 — Observations

`env.reset()` returns a **dictionary**, not an array. This is a deliberate design choice and it's the second-most-common source of beginner bugs, because almost every RL library expects a flat array.

### What's in the dict

```python
obs = env.reset()
for k, v in obs.items():
    print(k, np.array(v).shape)
```

Typical output for `Lift` with a Panda:

```
robot0_joint_pos_cos        (7,)     cos of each joint angle
robot0_joint_pos_sin        (7,)     sin of each joint angle
robot0_joint_vel            (7,)     joint angular velocities
robot0_eef_pos              (3,)     gripper position in world coords (metres)
robot0_eef_quat             (4,)     gripper orientation, quaternion
robot0_gripper_qpos         (2,)     finger positions
robot0_gripper_qvel         (2,)     finger velocities
cube_pos                    (3,)     object position
cube_quat                   (4,)     object orientation
gripper_to_cube_pos         (3,)     vector from gripper to object
object-state                (14,)    all object info concatenated
robot0_proprio-state        (32,)    all robot info concatenated
```

Two things worth noticing.

**Joint angles come as cos/sin, not raw radians.** Because an angle of 0 and an angle of 2π are the same configuration but numerically far apart. Feeding raw radians to a network creates a discontinuity that doesn't exist physically. Cos/sin removes it.

**`gripper_to_cube_pos` is a gift.** It's the relative vector — exactly the quantity the policy needs. Including relative positions rather than making the network subtract two absolute positions itself measurably speeds up learning.

### Choosing what to feed your agent

Two regimes, and they are genuinely different problems:

**State-based (start here).** Feed the numbers. Small networks, fast training, solvable on a laptop.

```python
OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat",
            "robot0_gripper_qpos", "object-state"]
```

**Vision-based.** Feed camera images. Realistic — a real robot doesn't get the object's exact coordinates handed to it — but far harder and far slower.

```python
env = suite.make(...,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=84, camera_widths=84,
)
```

> **Do state-based first.** Always. If your algorithm can't solve `Lift` from state, adding a CNN will not help — you'll just be debugging two problems at once, in ten times the wall-clock.

### The image is upside down

MuJoCo renders with the origin at the bottom-left; images are conventionally top-left. So:

```python
img = obs["agentview_image"]      # (H, W, 3), uint8, VERTICALLY FLIPPED
img = np.flipud(img)              # fix it
img = img.transpose(2, 0, 1)      # HWC -> CHW for PyTorch
img = np.ascontiguousarray(img)   # transpose leaves a non-contiguous view
```

Nothing will crash if you skip the flip. The agent will just learn from upside-down images, which works but makes your debug visualisations confusing and your saved videos wrong.

---

## Part 6 — Rewards

### Sparse vs shaped

```python
reward_shaping=False   # 1.0 on success, 0.0 otherwise
reward_shaping=True    # partial credit for partial progress
```

**Sparse is honest and nearly unlearnable from scratch.** A random policy will essentially never complete a pick-and-place by accident, so the reward is zero forever, so there is no gradient, so nothing learns. This isn't a tuning problem — it's a structural one.

**Shaped rewards give partial credit.** For `PickPlaceCan` the reward decomposes into stages:

```python
r_reach, r_grasp, r_lift, r_hover = env.staged_rewards()
```

| Stage | Meaning | Contribution |
|---|---|---|
| reach | gripper approaches the object | `(1 − tanh(10·d)) × 0.1` |
| grasp | fingers make contact and close | `0.35` |
| lift | object rises off the table | `0.35 → 0.5` |
| hover | object moves over the target bin | `0.5 → 0.7` |

The reach term is worth studying, because it's the template for shaped rewards generally:

```
r_reach = (1 − tanh(10 · distance)) × 0.1
```

At `d = 0` it's 0.1. At `d = 0.1 m` it's 0.024. At `d = 0.5 m` it's ~0.0001. It's a smooth gradient that always points toward the object, never zero, so there is *always* something to climb even when the policy is terrible.

**Use `reward_shaping=True` while learning.** Report sparse success rate as your actual metric. Those are different things and conflating them is a common way to overstate results.

### Recovering the true distance

The reach reward is invertible, which is the most reliable way to get a distance metric that's guaranteed consistent with the reward:

```python
import numpy as np

def distance_from_reach_reward(r_reach, mult=0.1):
    """Invert (1 - tanh(10d)) * mult to recover d in metres."""
    if r_reach <= 0:
        return float("inf")
    t = 1.0 - r_reach / mult
    t = min(max(t, 0.0), 1.0 - 1e-9)
    return float(np.arctanh(t) / 10.0)
```

This is more robust than indexing `sim.data.site_xpos` directly, where getting the wrong index silently produces a plausible-looking number that isn't a distance at all.

### Sanity-check the reward scale before training

```python
env = make_env()          # from Part 7, or any suite.make(...) you like
rewards = []
for ep in range(5):
    env.reset()
    for t in range(200):
        _, r, done, _ = env.step(np.random.uniform(*env.action_spec))
        rewards.append(r)
        if done: break

r = np.array(rewards)
print(f"min {r.min():.5f}  max {r.max():.5f}  mean {r.mean():.5f}")
print(f"fraction above 0.1: {(r > 0.1).mean():.4f}")
```

If `max` is tiny and the fraction is 0.0000, random exploration is never reaching the interesting states. That tells you something important *before* you spend a day training: you may need a longer horizon, an easier task, or demonstrations.

---

## Part 7 — The wrapper you must write

Robosuite returns dicts. RL algorithms want arrays. Bridge them yourself — it's twenty lines and you'll understand exactly what your agent sees, which you won't if you use a black-box wrapper.

```python
import numpy as np

class RobosuiteWrapper:
    """Dict observations -> flat float32 array. Gym-style interface."""

    def __init__(self, env, obs_keys):
        self.env = env
        self.obs_keys = obs_keys
        self.low, self.high = env.action_spec
        self.act_dim = self.low.shape[0]
        obs = self.reset()
        self.obs_dim = obs.shape[0]

    def _flatten(self, obs_dict):
        parts = [np.asarray(obs_dict[k], dtype=np.float32).ravel()
                 for k in self.obs_keys if k in obs_dict]
        return np.concatenate(parts)

    def reset(self):
        return self._flatten(self.env.reset())

    def step(self, action):
        a = np.clip(action, self.low, self.high)
        obs, reward, done, info = self.env.step(a)
        return self._flatten(obs), float(reward), bool(done), info

    def sample_action(self):
        return np.random.uniform(self.low, self.high).astype(np.float32)

    def close(self):
        self.env.close()
```

Three details that matter more than they look:

**Filter with `if k in obs_dict`.** Observation keys differ between tasks — `Lift` has `cube_pos`, `PickPlaceCan` has `Can_pos`. Filtering means the same wrapper works across tasks instead of crashing.

**Derive `obs_dim` by actually resetting.** Do not compute it by hand. A hardcoded dimension that silently disagrees with reality is one of the nastiest bugs in this space: it surfaces much later as a shape mismatch when loading a checkpoint, long after you've forgotten what you changed.

**Keep the key order fixed.** `self.obs_keys` is a list, and concatenation order defines what each network input index means. Change the order between training and evaluation and your policy sees scrambled input while everything still runs.

> Robosuite does ship a `GymWrapper`, and 1.5 <cite index="34-1">added GymWrapper support for both gym and gymnasium with dict observations.</cite> Use it once you know what it's doing. Write your own first.

---

## Part 8 — Choosing an algorithm

You now have a working environment. What do you run on it?

| Algorithm | Sample efficiency | Stability | Verdict for robosuite |
|---|---|---|---|
| **SAC** | High | High | **Start here.** Off-policy, handles continuous actions natively, auto-tunes exploration. |
| TD3 | High | Medium | Fine. Needs manual exploration-noise tuning. |
| PPO | Low | Very high | Very stable but needs 10–50× more environment steps. Painful when each step is a physics sim. |
| DrQ-v2 | High | Medium | The choice for *image* observations. |
| DreamerV3 | Very high | Medium | Model-based. Best sample efficiency, most complex. Advanced. |

**The reasoning:** every environment step costs real physics computation. Robosuite runs roughly 500–2000 steps/second on one CPU core. PPO needing 10 million steps means hours-to-days; SAC solving `Lift` in 100k steps means under an hour.

So: **off-policy, replay-buffer methods.** SAC is the default, and it's what we'll build.

---

## Part 9 — SAC, explained intuitively

Three ideas. Learn these and the code writes itself.

### Idea 1: the critic learns to predict, the actor learns to exploit

The **critic** `Q(s, a)` answers: *if I'm in state s and take action a, how much total future reward should I expect?* That's a supervised regression problem — you get a target from the Bellman equation.

The **actor** `π(s)` answers: *what action should I take?* There's no supervised target; you only find out by trying. So the actor is trained to output actions the critic scores highly. The easy problem bootstraps the hard one.

### Idea 2: maximum entropy — be as random as you can afford to be

Standard RL maximises reward. SAC maximises reward **plus the entropy of the policy**:

```
J = E[ Σ  r  +  α · H(π(·|s)) ]
```

Read that as: *get reward, but stay as unpredictable as you can while doing it.*

Why this is not just an exploration hack. It prevents the policy collapsing onto the first mediocre strategy it finds. In manipulation there are many ways to grasp an object; a policy that commits early to one is stuck if that one turns out to be fragile. The entropy term keeps alternatives alive until the reward evidence is strong enough to abandon them.

`α` is the temperature — how much you value randomness. Setting it by hand is miserable, so SAC learns it: pick a **target entropy** (convention: `−act_dim`) and adjust `α` to hit it. Too random → shrink α. Too deterministic → grow α.

### Idea 3: two critics, and take the minimum

Q-learning systematically **over**estimates. The target uses `max`/near-max over actions, and the max of noisy estimates is biased upward. That bias compounds through bootstrapping — the critic hallucinates value, the actor chases it, performance collapses.

The fix is almost silly: train **two independent critics** and use the smaller of the two.

```python
q_target = torch.min(q1_target, q2_target)
```

Two independent estimates rarely overestimate in the same place, so the minimum cancels most of the bias. This one line is a large fraction of why SAC is stable.

### Squashed Gaussian and its log-prob correction

Actions must lie in `[-1, 1]`, so SAC samples from a Gaussian and squashes with `tanh`:

```text
u ~ Normal(μ, σ)
a = tanh(u)
```

Squashing changes the probability density, so the log-prob needs a change-of-variables correction:

```text
log π(a) = log N(u) − Σ log(1 − tanh²(u))
```

Computed naively this explodes as `a → ±1`, because `log(1 − a²) → −∞`. The numerically stable form is standard and worth copying exactly:

```python
logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
```

If you write the naive version you get NaNs after a few thousand steps, and the cause is genuinely hard to find. Use this line.

---

## Part 10 — The full implementation

This code is tested — it learns a continuous-control task from scratch.

### `sac.py`

```python
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG_STD_MIN, LOG_STD_MAX = -20, 2


def mlp(sizes, act=nn.ReLU, out_act=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1]),
                   act() if i < len(sizes) - 2 else out_act()]
    return nn.Sequential(*layers)


class SquashedGaussianActor(nn.Module):
    """Outputs a tanh-squashed Gaussian over actions in [-1, 1]."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.net = mlp([obs_dim, *hidden], act=nn.ReLU, out_act=nn.ReLU)
        self.mu = nn.Linear(hidden[-1], act_dim)
        self.log_std = nn.Linear(hidden[-1], act_dim)

    def forward(self, obs, deterministic=False, with_logprob=True):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        dist = torch.distributions.Normal(mu, std)
        # rsample = reparameterised: gradients flow through the sample
        u = mu if deterministic else dist.rsample()
        a = torch.tanh(u)

        if with_logprob:
            logp = dist.log_prob(u).sum(-1)
            # numerically stable tanh change-of-variables correction
            logp -= (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(-1)
        else:
            logp = None
        return a, logp


class Critic(nn.Module):
    """Twin Q networks. The minimum of the two fights overestimation."""

    def __init__(self, obs_dim, act_dim, hidden=(256, 256)):
        super().__init__()
        self.q1 = mlp([obs_dim + act_dim, *hidden, 1])
        self.q2 = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class Replay:
    """Flat transition buffer. Off-policy methods reuse every transition."""

    def __init__(self, cap, obs_dim, act_dim):
        self.o = np.zeros((cap, obs_dim), np.float32)
        self.o2 = np.zeros((cap, obs_dim), np.float32)
        self.a = np.zeros((cap, act_dim), np.float32)
        self.r = np.zeros(cap, np.float32)
        self.d = np.zeros(cap, np.float32)
        self.ptr, self.size, self.cap = 0, 0, cap

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.o[i], self.a[i], self.r[i], self.o2[i], self.d[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, n, dev):
        idx = np.random.randint(0, self.size, n)
        t = lambda x: torch.as_tensor(x[idx], device=dev)
        return t(self.o), t(self.a), t(self.r), t(self.o2), t(self.d)


class SAC:
    def __init__(self, obs_dim, act_dim, device="cpu",
                 lr=3e-4, gamma=0.99, tau=0.005,
                 alpha_init=0.2, auto_alpha=True):
        self.dev = torch.device(device)
        self.actor = SquashedGaussianActor(obs_dim, act_dim).to(self.dev)
        self.critic = Critic(obs_dim, act_dim).to(self.dev)
        self.critic_targ = Critic(obs_dim, act_dim).to(self.dev)
        self.critic_targ.load_state_dict(self.critic.state_dict())
        for p in self.critic_targ.parameters():
            p.requires_grad_(False)

        self.pi_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.q_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma, self.tau = gamma, tau

        self.auto_alpha = auto_alpha
        if auto_alpha:
            # convention: target entropy = -act_dim
            self.target_entropy = -float(act_dim)
            self.log_alpha = torch.tensor(np.log(alpha_init),
                                          requires_grad=True, device=self.dev)
            self.a_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        else:
            self.log_alpha = torch.tensor(np.log(alpha_init), device=self.dev)

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        o = torch.as_tensor(obs, dtype=torch.float32,
                            device=self.dev).unsqueeze(0)
        a, _ = self.actor(o, deterministic, with_logprob=False)
        return a.squeeze(0).cpu().numpy()

    def update(self, batch):
        o, a, r, o2, d = batch

        # ---- critic: regress onto the entropy-augmented Bellman target ----
        with torch.no_grad():
            a2, logp2 = self.actor(o2)
            q1t, q2t = self.critic_targ(o2, a2)
            qt = torch.min(q1t, q2t) - self.alpha * logp2   # min = anti-overest.
            backup = r + self.gamma * (1 - d) * qt

        q1, q2 = self.critic(o, a)
        q_loss = F.mse_loss(q1, backup) + F.mse_loss(q2, backup)
        self.q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_opt.step()

        # ---- actor: maximise Q, minus the entropy price ----
        for p in self.critic.parameters():
            p.requires_grad_(False)          # don't build critic grads here
        pi, logp = self.actor(o)
        q1p, q2p = self.critic(o, pi)
        pi_loss = (self.alpha * logp - torch.min(q1p, q2p)).mean()
        self.pi_opt.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_opt.step()
        for p in self.critic.parameters():
            p.requires_grad_(True)

        # ---- temperature: steer entropy toward the target ----
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha
                           * (logp.detach() + self.target_entropy)).mean()
            self.a_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.a_opt.step()

        # ---- soft target update: slow-moving copy stabilises the target ----
        with torch.no_grad():
            for pt, p in zip(self.critic_targ.parameters(),
                             self.critic.parameters()):
                pt.data.mul_(1 - self.tau).add_(p.data, alpha=self.tau)

        return dict(q=q_loss.item(), pi=pi_loss.item(),
                    alpha=self.alpha.item(), entropy=-logp.mean().item())
```

### `train.py`

```python
import os
os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from sac import SAC, Replay
# from wrapper import RobosuiteWrapper   (Part 7)

OBS_KEYS = ["robot0_eef_pos", "robot0_eef_quat",
            "robot0_gripper_qpos", "object-state"]

TOTAL_STEPS = 300_000
START_STEPS = 5_000      # uniform random actions before the policy takes over
BATCH_SIZE = 256
UPDATES_PER_STEP = 1
EVAL_EVERY = 10_000


def make_env(task="Lift", horizon=200):
    raw = suite.make(
        env_name=task, robots="Panda",
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, reward_shaping=True,
        control_freq=20, horizon=horizon, ignore_done=False,
        controller_configs=load_composite_controller_config(
            controller=None, robot="Panda"),
    )
    return RobosuiteWrapper(raw, OBS_KEYS)


def evaluate(agent, env, n=5):
    """Deterministic rollouts. This is your real performance number."""
    returns = []
    for _ in range(n):
        o, ret, done = env.reset(), 0.0, False
        while not done:
            o, r, done, _ = env.step(agent.act(o, deterministic=True))
            ret += r
        returns.append(ret)
    return float(np.mean(returns))


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    env = make_env()
    eval_env = make_env()
    print(f"obs_dim={env.obs_dim}  act_dim={env.act_dim}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SAC(env.obs_dim, env.act_dim, device=dev)
    buf = Replay(500_000, env.obs_dim, env.act_dim)

    o, ep_ret, ep_len, returns = env.reset(), 0.0, 0, []

    for step in range(1, TOTAL_STEPS + 1):
        a = env.sample_action() if step < START_STEPS else agent.act(o)
        o2, r, done, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        # Time-limit truncation is NOT a real terminal state. Bootstrapping
        # through it is correct; treating it as terminal teaches the agent
        # the world ends at step 200.
        timeout = ep_len >= env.env.horizon
        buf.add(o, a, r, o2, float(done and not timeout))
        o = o2

        if done:
            returns.append(ep_ret)
            o, ep_ret, ep_len = env.reset(), 0.0, 0

        if step >= START_STEPS:
            for _ in range(UPDATES_PER_STEP):
                info = agent.update(buf.sample(BATCH_SIZE, agent.dev))

        if step % 1000 == 0 and returns:
            print(f"step {step:>7,}  train_return(20) {np.mean(returns[-20:]):7.3f}"
                  f"  q {info['q']:.3f}  alpha {info['alpha']:.3f}"
                  f"  H {info['entropy']:+.2f}")

        if step % EVAL_EVERY == 0 and step >= START_STEPS:
            score = evaluate(agent, eval_env)
            print(f"  === EVAL @ {step:,}: {score:.3f} ===")
            torch.save({"actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                        "step": step}, "sac_checkpoint.pt")

    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
```

### The time-limit detail, because it bites everyone

```python
timeout = ep_len >= env.env.horizon
buf.add(o, a, r, o2, float(done and not timeout))
```

`done` is `True` both when the task genuinely ends **and** when the episode is simply cut off at the horizon. These are completely different situations.

A real terminal means no future reward exists. A timeout means future reward exists, you just stopped watching. If you store timeouts as terminals, the Bellman target sets future value to zero for those states, and the agent learns that the universe ends at step 200. It becomes short-sighted in a way that's very hard to diagnose from the loss curves.

Store `done=0` for timeouts. Two lines, large effect.

---

## Part 11 — Debugging checklist

In the order things actually go wrong.

### Before you train

**1. Can random actions get any reward at all?**
```python
print(f"max reward seen: {max(rewards):.4f}")
```
If it's ~0 across a few thousand random steps, no algorithm will bootstrap from this. Change the task, lengthen the horizon, or get demonstrations.

**2. Does your observation dimension match between train and eval?**
Print it in both. A mismatch surfaces as a shape error at checkpoint load, hours later.

**3. Is your action range right?**
```python
a = agent.act(o)
assert np.all(np.abs(a) <= 1.0), f"policy output out of range: {a}"
```

### During training

| Symptom | Likely cause |
|---|---|
| Q-loss exploding to 1e6+ | Overestimation. Check you're using `min(q1, q2)`, not `q1`. |
| Return flat at exactly 0 | Reward never fires. Check `reward_shaping=True`. |
| NaN after a few thousand steps | The log-prob correction. Use the `softplus` form. |
| Entropy → very negative | α collapsed. Check the alpha loss sign. |
| Agent good in training, bad in eval | You forgot `deterministic=True` at eval. |
| Learns then collapses | Learning rate too high, or replay too small. |
| Nothing happens for 50k steps | Often normal. Compare against a random-policy baseline before panicking. |

### The single most useful habit

**Log a random-policy baseline and plot it as a flat line on every chart.** Then "is my agent learning?" becomes a question you can answer at a glance rather than a feeling. A surprising number of published-looking curves sit below their own random baseline.

### The second most useful habit

**Watch a rollout.** Numbers hide things video doesn't:

```python
env = suite.make(..., has_renderer=True)
o = env.reset()
for _ in range(200):
    o, r, done, _ = env.step(agent.act(o, deterministic=True))
    env.render()
    if done: break
```

Ten seconds of footage will tell you the gripper never closes, or the arm is stuck against the table, or it's hovering 2 cm above the object forever. None of that is visible in a loss curve.

---

## Part 12 — Where to go next

**Once `Lift` works from state:**

1. **Harder tasks.** `Door` → `Stack` → `PickPlaceCan`. Change one thing at a time.
2. **Vision.** Turn on `use_camera_obs`, add a CNN encoder, switch to DrQ-v2 (SAC plus random-shift augmentation). Expect 5–10× more steps.
3. **Demonstrations.** Robomimic ships human demos for robosuite tasks. Seeding the replay buffer with them turns an unlearnable sparse-reward problem into a learnable one, and it is the standard answer when exploration stalls.
4. **Model-based RL.** DreamerV3 learns a world model and trains the policy inside it — dramatically better sample efficiency at the cost of significant implementation complexity. Go here after you're comfortable with SAC, not before.

### Suggested reading order

- Robosuite docs — Environments, then Controllers
- SAC paper (Haarnoja et al. 2018), sections 4–5
- Robomimic paper — an excellent study of what actually matters in manipulation learning

---

## Appendix — Minimum viable starting config

```text
TASK        = "Lift"          # start easy, always
ROBOT       = "Panda"
CONTROLLER  = "OSC_POSE"      # via load_composite_controller_config
CONTROL_FREQ = 20
HORIZON     = 200
REWARD_SHAPING = True
OBS         = state-based     # no images until state works

ALGORITHM   = SAC
LR          = 3e-4
BATCH       = 256
GAMMA       = 0.99
TAU         = 0.005
BUFFER      = 500_000
START_STEPS = 5_000
UPDATES/STEP = 1
```

Expect `Lift` to show clear progress within roughly 50k–100k environment steps. If nothing has moved by 150k, something is wrong — go back to Part 11 rather than waiting longer.

---

*Everything here has been checked against robosuite 1.5. If you're on 1.4, the controller loading API differs — see the version warning in Part 4.*