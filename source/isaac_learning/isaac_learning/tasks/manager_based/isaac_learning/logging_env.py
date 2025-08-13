# source/isaac_learning/tasks/manager_based/franka/franka_reach_env.py

from isaaclab.envs import ManagerBasedRLEnv
import random
import torch
import numpy as np
from isaaclab.utils.math import quat_from_euler_xyz


class FrankaReachEnv(ManagerBasedRLEnv):
    """
    Custom environment for the Franka reach task that includes detailed
    logging of individual reward terms.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        # cache prim paths & initial threshold
        self.goal_path = self.cfg.sim.scene.ee_goal.prim_path
        self.shelf_path = self.cfg.sim.scene.shelf.prim_path
        self.success_thr = self.cfg.task.goal_pose_range.get("threshold", 0.05)
        self.goal_ranges = self.cfg.task.goal_pose_range
        self.shelf_ranges = self.cfg.task.shelf_pose_range

    def _reset_idx(self, env_ids: np.ndarray):
        super()._reset_idx(env_ids)
        for i in env_ids:
            # sample & set goal
            gx, gy, gz = [self._np_random.uniform(*self.goal_ranges[k]) for k in ("x", "y", "z")]
            gr = random.choice(self.goal_ranges["rpy"])
            self.scene.set_world_pose(self.goal_path,
                                      position=(gx, gy, gz),
                                      orientation=quat_from_euler_xyz(*gr),
                                      env_index=i)
            # sample & set shelf
            sx, sy, sz = [self._np_random.uniform(*self.shelf_ranges[k]) for k in ("x", "y", "z")]
            sr = random.choice(self.shelf_ranges["rpy"])
            self.scene.set_world_pose(self.shelf_path,
                                      position=(sx, sy, sz),
                                      orientation=quat_from_euler_xyz(*sr),
                                      env_index=i)

            robot = env.scene["robot"]
            # 1) root pose (offset by env origins if you use them)
            root_state = robot.data.default_root_state.clone()
            root_state[:, :3] += env.scene.env_origins   # if you use InteractiveScene origins
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids=env_ids)
            # 2) zero root velocity
            robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]), env_ids=env_ids)
            # 3) joint state with zero dq
            q = robot.data.default_joint_pos.clone()
            dq = torch.zeros_like(robot.data.default_joint_vel)
            robot.write_joint_state_to_sim(q, dq, env_ids=env_ids)
            # 4) finalize reset (clears internal buffers)
            robot.reset(env_ids=env_ids)
            print("RESET")

    def _get_extras(self):
        extras = super()._get_extras()
        # log individual reward terms
        for name, val in self.reward_manager.component_reward_terms.items():
            extras[f"Rewards/{name}"] = val.mean()
        # compute success & distance
        robot = self.scene.robots[0]
        idx = robot.body_names.index("panda_hand")
        ee = robot.data.body_pos_w[:, idx]
        tgt = self.scene.ee_goal.data.target_pos_w[:, 0, :]
        dist = torch.norm(ee - tgt, dim=1)
        extras["is_success"] = (dist < self.success_thr).float()
        extras["dist_to_target"] = dist
        return extras
