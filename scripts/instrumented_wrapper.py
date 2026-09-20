
"""
Paste this file into /content/dreamerv3_panda/utils/instrumented_wrapper.py

It replaces your existing RobosuiteWrapper with an instrumented version
that extracts all five metrics from robosuite's internal state every step:

  1. reach_success  — gripper within 5cm of can
  2. grasp_success  — fingers in contact with can
  3. lift_success   — can lifted above table (grasped + elevated)
  4. place_success  — can inside target bin
  5. dist_to_object — raw Euclidean distance, gripper tip to can centre

These are reported:
  - Per step in the CSV log (for plotting continuous learning curves)
  - Per episode (aggregated as: did any step in the episode achieve this?)
  - In the console log every log_every steps

The episode-level success flags give you the clearest dissertation metrics:
  grasp_rate  = episodes where grasp_success was ever True / total episodes
  lift_rate   = episodes where lift_success was ever True / total episodes
  place_rate  = episodes where place_success was ever True / total episodes
"""

import numpy as np


class InstrumentedWrapper:
    """
    Drop-in replacement for RobosuiteWrapper.
    All existing code (buf.add, env.step, env.reset) works identically.

    Extra: call env.get_metrics() after env.step() to get the 5-tuple.
    Extra: call env.episode_metrics() at episode end to get episode stats.
    """

    def __init__(self, env, cfg):
        self.env  = env
        self.cfg  = cfg
        self._low, self._high = env.action_spec
        self.act_dim = self._low.shape[0]

        # Per-step metric (updated after every step)
        self._metrics = {
            "reach_success":  False,
            "grasp_success":  False,
            "lift_success":   False,
            "place_success":  False,
            "dist_to_object": float("inf"),
        }

        # Per-episode accumulators
        self._ep_any_reach = False
        self._ep_any_grasp = False
        self._ep_any_lift  = False
        self._ep_any_place = False
        self._ep_min_dist  = float("inf")
        self._ep_steps     = 0

    # ── Public API ───────────────────────────────────────────────────────────

    def sample_action(self):
        return np.random.uniform(self._low, self._high).astype(np.float32)

    def reset(self):
        self._ep_any_reach = False
        self._ep_any_grasp = False
        self._ep_any_lift  = False
        self._ep_any_place = False
        self._ep_min_dist  = float("inf")
        self._ep_steps     = 0
        return self._process(self.env.reset())

    def step(self, action):
        action = np.clip(action.astype(np.float64), self._low, self._high)
        obs_dict, reward, done, info = self.env.step(action)
        self._update_metrics()
        self._ep_steps += 1
        return self._process(obs_dict), float(reward), bool(done), info

    def get_metrics(self):
        """Return per-step metrics dict after the most recent step."""
        return dict(self._metrics)

    def episode_summary(self):
        """
        Return per-episode aggregate metrics.
        Call this when done=True before calling reset().

        Returns dict with:
          ep_reach_success  bool   — gripper got within 5cm at any point
          ep_grasp_success  bool   — grasped can at any point
          ep_lift_success   bool   — lifted can at any point
          ep_place_success  bool   — placed can in bin (task complete)
          ep_min_dist       float  — closest the gripper got to the can
          ep_steps          int    — total steps in episode
          ep_stage          str    — highest stage reached this episode
        """
        if self._ep_any_place:
            stage = "place"
        elif self._ep_any_lift:
            stage = "lift"
        elif self._ep_any_grasp:
            stage = "grasp"
        elif self._ep_any_reach:
            stage = "reach"
        else:
            stage = "none"

        return {
            "ep_reach_success": self._ep_any_reach,
            "ep_grasp_success": self._ep_any_grasp,
            "ep_lift_success":  self._ep_any_lift,
            "ep_place_success": self._ep_any_place,
            "ep_min_dist":      round(self._ep_min_dist, 4),
            "ep_steps":         self._ep_steps,
            "ep_stage":         stage,
        }

    def close(self):
        self.env.close()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _update_metrics(self):
        """
        Extract all 5 metrics from robosuite's internal simulation state.
        Called after every env.step().

        How each metric is computed:
          dist_to_object  — directly from sim: eef_site_xpos vs obj body_xpos
          reach_success   — dist < 0.05m (5cm threshold, same scale as tanh reward)
          grasp_success   — staged_rewards()[1] > 0 (robosuite's own grasp check)
          lift_success    — staged_rewards()[2] > staged_rewards()[1]
                            (lift reward > grasp reward = can is elevating)
          place_success   — env.objects_in_bins[0] > 0 (robosuite's placement check)
        """
        raw_env = self.env  # the underlying robosuite env

        try:
            # ── Distance to object ────────────────────────────────────────────
            # Get can body position from sim
            can_name = raw_env.objects[0].name
            can_pos  = raw_env.sim.data.body_xpos[raw_env.obj_body_id[can_name]]

            # Get gripper tip position (eef_site)
            # robots[0].arms returns ['right'] for Panda
            arm      = raw_env.robots[0].arms[0]
            eef_site = raw_env.robots[0].eef_site_id[arm]
            eef_pos  = raw_env.sim.data.site_xpos[eef_site]

            dist = float(np.linalg.norm(eef_pos - can_pos))

            # ── Stage rewards (gives us grasp + lift booleans) ────────────────
            # Call robosuite's own staged_rewards() — returns (reach, grasp, lift, hover)
            r_reach, r_grasp, r_lift, r_hover = raw_env.staged_rewards()

            # ── Derive boolean flags ──────────────────────────────────────────
            reach_success = dist < 0.05            # within 5cm
            grasp_success = r_grasp > 0.0          # binary: contact with can
            lift_success  = r_lift > r_grasp       # lift reward exceeds grasp base
            place_success = bool(
                hasattr(raw_env, "objects_in_bins") and
                raw_env.objects_in_bins[0] > 0
            )

            # Store per-step
            self._metrics = {
                "reach_success":  reach_success,
                "grasp_success":  grasp_success,
                "lift_success":   lift_success,
                "place_success":  place_success,
                "dist_to_object": round(dist, 4),
            }

            # Accumulate episode bests
            self._ep_any_reach = self._ep_any_reach or reach_success
            self._ep_any_grasp = self._ep_any_grasp or grasp_success
            self._ep_any_lift  = self._ep_any_lift  or lift_success
            self._ep_any_place = self._ep_any_place or place_success
            self._ep_min_dist  = min(self._ep_min_dist, dist)

        except Exception as e:
            # Fail silently — never break training due to metric extraction
            pass

    def _process(self, obs_dict):
        cam  = self.cfg["env"]["camera"] + "_image"
        img  = np.ascontiguousarray(np.flipud(obs_dict[cam])).transpose(2, 0, 1)
        parts = [
            np.asarray(obs_dict[k], np.float32).flatten()
            for k in self.cfg["env"]["obs_keys"] if k in obs_dict
        ]
        vec  = np.concatenate(parts) if parts else np.zeros(1, np.float32)
        return {"image": img.copy(), "vector": vec}

