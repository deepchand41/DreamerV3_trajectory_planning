"""
envs/panda_env.py

Robosuite Panda environment wrapped as a standard Gym env.
Handles:
  - Multi-modal observations (image + proprioception vector)
  - OSC_POSE controller
  - Dense reward shaping
  - Headless rendering via OSMesa
"""

import os
import numpy as np
import gym
from gym import spaces

# Force OSMesa before any MuJoCo import
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import load_controller_config


# ─────────────────────────────────────────────────────────────────────────────
# Core Robosuite → Gym wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PandaPickPlaceEnv(gym.Env):
    """
    Robosuite PickPlaceCan task on the Panda arm, wrapped as a Gym env.

    Observation space (Dict):
      image  : (3, H, W) uint8  — agentview RGB camera
      vector : (D,) float32     — concatenated proprioception + object state

    Action space:
      Box(7,) — OSC_POSE: [dx, dy, dz, dRx, dRy, dRz, gripper]
                normalised to [-1, +1]
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, config):
        super().__init__()
        self.cfg = config
        img_size  = config["image_size"]   # e.g. 64
        self._horizon = config.get("horizon", 500)

        # ── Build Robosuite env ───────────────────────────────────────────
        controller_cfg = load_controller_config(
            default_controller=config.get("controller", "OSC_POSE")
        )
        self._env = suite.make(
            env_name=config["task"],         # "PickPlaceCan"
            robots=config.get("robot", "Panda"),
            has_renderer=False,              # no on-screen render
            has_offscreen_renderer=True,     # needed for camera obs
            use_camera_obs=True,
            camera_names=config.get("camera", "agentview"),
            camera_heights=img_size,
            camera_widths=img_size,
            reward_shaping=config.get("reward_shaping", True),
            control_freq=config.get("control_freq", 20),
            horizon=self._horizon,
            ignore_done=False,
            controller_configs=controller_cfg,
        )

        # ── Determine vector obs dimension ────────────────────────────────
        raw_obs = self._env.reset()
        vec = self._build_vector(raw_obs)
        vec_dim = vec.shape[0]

        # ── Define spaces ─────────────────────────────────────────────────
        self.observation_space = spaces.Dict({
            "image":  spaces.Box(0, 255, (3, img_size, img_size), dtype=np.uint8),
            "vector": spaces.Box(-np.inf, np.inf, (vec_dim,), dtype=np.float32),
        })
        # OSC_POSE: 6 DOF Cartesian delta + 1 gripper
        self.action_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)

        self._step_count = 0

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_image(self, obs_dict):
        """Extract camera image and convert HWC uint8 → CHW uint8."""
        cam_key = self.cfg.get("camera", "agentview") + "_image"
        img = obs_dict[cam_key]              # (H, W, 3) uint8
        img = np.flipud(img)                 # robosuite renders upside-down
        return img.transpose(2, 0, 1)        # HWC → CHW

    def _build_vector(self, obs_dict):
        """Concatenate requested proprioceptive / object-state keys."""
        keys = self.cfg.get("obs_keys", [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "robot0_joint_pos",
            "object-state",
        ])
        parts = []
        for k in keys:
            if k in obs_dict:
                v = obs_dict[k]
                parts.append(np.asarray(v, dtype=np.float32).flatten())
        return np.concatenate(parts, axis=0) if parts else np.zeros(1, dtype=np.float32)

    def _process_obs(self, obs_dict):
        return {
            "image":  self._build_image(obs_dict),
            "vector": self._build_vector(obs_dict),
        }

    # ── Gym API ───────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        self._step_count = 0
        obs_dict = self._env.reset()
        return self._process_obs(obs_dict), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float64)
        obs_dict, reward, done, info = self._env.step(action)
        self._step_count += 1
        truncated = self._step_count >= self._horizon
        terminated = done and not truncated
        processed  = self._process_obs(obs_dict)
        return processed, float(reward), terminated, truncated, info

    def render(self, mode="rgb_array"):
        img = self._env.sim.render(
            height=self.cfg["image_size"],
            width=self.cfg["image_size"],
            camera_name=self.cfg.get("camera", "agentview"),
        )
        return np.flipud(img)

    def close(self):
        self._env.close()


# ─────────────────────────────────────────────────────────────────────────────
# Factory helper
# ─────────────────────────────────────────────────────────────────────────────

def make_env(config: dict) -> PandaPickPlaceEnv:
    """Convenience factory used by training scripts."""
    return PandaPickPlaceEnv(config)