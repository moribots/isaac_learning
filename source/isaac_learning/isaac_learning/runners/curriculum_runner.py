from rsl_rl.runners import OnPolicyRunner
import os
import glob
import wandb


class CurriculumRunner(OnPolicyRunner):
    """
    Adjusts env parameters based on observed success rate.
    Also uploads any newly created training videos to W&B.
    """

    def __init__(self, *args, upload_every: int = 200, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_dir = os.path.join(self.log_dir, "videos", "train")
        self._logged_videos = set()
        self.upload_every = int(upload_every)
        self._update_i = 0

    def _log_new_videos(self):
        if not os.path.isdir(self._video_dir):
            print(f"[WARNING] Video directory {self._video_dir} does not exist. Skipping video upload.")
            return
        for path in glob.glob(os.path.join(self._video_dir, "**", "*.mp4"), recursive=True):
            if path in self._logged_videos:
                continue
            wandb.log({"train/policy_video": wandb.Video(path, fps=30)}, commit=False)
            self._logged_videos.add(path)

    def after_update(self, metrics: dict):
        super().after_update(metrics)
        # Curriculum based on success rate already present
        # succ = metrics.get("train/is_success", 0.0)
        # cm = self.task.env.curriculum_manager
        # if succ > 0.9:
        #     cm.set_env_param("terminations.goal_reached.params.threshold", 0.01)
        # elif succ > 0.8:
        #     cm.set_env_param("terminations.goal_reached.params.threshold", 0.02)
        # else:
        #     cm.set_env_param("terminations.goal_reached.params.threshold", 0.05)
        # if succ > 0.85:
        #     cm.set_env_param("rewards.collision_penalty.weight", -100.0)

        # Log scalar rewards already in your file
        try:
            extras = self.task.env.get_attr("extras")[0]
            for key, val in extras.items():
                if key.startswith("Rewards/") or key == "is_success":
                    wandb.log({f"train/{key}": val}, commit=False)
        except Exception:
            print("[WARNING] No extras found in the environment. Skipping reward logging.")

        # Upload video
        self._update_i += 1
        if self._update_i % self.upload_every == 0:
            print(f"[INFO] Uploading videos to W&B...")
            self._log_new_videos()
            wandb.log({}, commit=True)
