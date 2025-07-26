# source/isaac_learning/tasks/manager_based/franka/franka_reach_env.py

from isaaclab.envs import ManagerBasedRLEnv


class FrankaReachEnv(ManagerBasedRLEnv):
    """
    Custom environment for the Franka reach task that includes detailed
    logging of individual reward terms.
    """

    def _get_rewards(self):
        """Calculates the rewards and populates the extras dict for logging."""
        # Let the parent class calculate the total reward first.
        super()._get_rewards()

        # Now, iterate through the reward terms and add their mean
        # values to the 'extras' dictionary. The RSL-RL logger
        # will automatically pick these up and send them to WandB.
        for term_name, term_value in self.reward_manager.component_reward_terms.items():
            self.extras[f"Rewards/{term_name}"] = term_value.mean()
