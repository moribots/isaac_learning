from rsl_rl.runners import OnPolicyRunner


class CurriculumRunner(OnPolicyRunner):
    """
    Adjusts env parameters based on observed success rate.
    """

    def after_update(self, metrics: dict):
        super().after_update(metrics)
        succ = metrics.get("train/is_success", 0.0)
        cm = self.task.env.curriculum_manager
        # tighten threshold as performance improves
        if succ > 0.9:
            cm.set_env_param("terminations.goal_reached.params.threshold", 0.01)
        elif succ > 0.8:
            cm.set_env_param("terminations.goal_reached.params.threshold", 0.02)
        else:
            cm.set_env_param("terminations.goal_reached.params.threshold", 0.05)
        # enable strong collision penalty once proficient
        if succ > 0.85:
            cm.set_env_param("rewards.collision_penalty.weight", -100.0)
