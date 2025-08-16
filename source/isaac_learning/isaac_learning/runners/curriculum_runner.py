from rsl_rl.runners import OnPolicyRunner
import os
import glob
import time
import sys
import wandb


class CurriculumRunner(OnPolicyRunner):
    def __init__(self, env, cfg, log_dir=None, device="cuda:0", upload_every=200):
        super().__init__(env, cfg, log_dir, device)
        self._video_dir = os.path.join(self.log_dir, "videos", "train")
        self._logged_videos = set()
        self.upload_every = int(upload_every)
        self._update_i = 0

        # Hook PPO updates
        _base_update = self.alg.update

        def _hooked_update(*args, **kwargs):
            out = _base_update(*args, **kwargs)
            self.after_update(self._collect_metrics())
            return out
        self.alg.update = _hooked_update

    def _collect_metrics(self) -> dict:
        rec = getattr(self.env.unwrapped, "recorder_manager", None)
        if rec is None:
            return {}
        succ = int(getattr(rec, "exported_successful_episode_count", 0))
        fail = int(getattr(rec, "exported_failed_episode_count", 0))
        tot = succ + fail
        return {"train/is_success": float(succ / tot) if tot > 0 else 0.0}

    def _fatal(self, msg: str, code: int = 2):
        print(f"[FATAL] {msg}")
        try:
            if wandb.run is not None:
                wandb.alert(title="Video upload failed", text=msg, level=wandb.AlertLevel.ERROR)
                wandb.finish()
        except Exception:
            pass
        sys.exit(code)

    def _log_new_videos_or_exit(self):
        # 1) Video directory must exist
        if not os.path.isdir(self._video_dir):
            self._fatal(f"Video directory missing: {self._video_dir}")

        # 2) Find candidate mp4s
        paths = sorted(glob.glob(os.path.join(self._video_dir, "**", "*.mp4"), recursive=True))
        if not paths:
            self._fatal(f"No MP4 files found under {self._video_dir}")

        # 3) Filter out zero-sized or already-logged files
        candidates = []
        for p in paths:
            try:
                if os.path.getsize(p) < 1024:  # skip tiny/incomplete
                    continue
            except OSError:
                continue
            if p in self._logged_videos:
                continue
            candidates.append(p)

        if not candidates:
            self._fatal("No new, non-empty MP4s to upload (all missing, tiny, or already logged).")

        # 4) Upload all candidates
        uploaded = 0
        for p in candidates:
            # small delay to ensure file is closed on slower FS
            time.sleep(0.01)
            wandb.log({"train/policy_video": wandb.Video(p, fps=30)}, step=self._update_i, commit=False)
            self._logged_videos.add(p)
            uploaded += 1

        if uploaded == 0:
            self._fatal("Scan yielded zero uploads after filtering.")

        wandb.log({"train/video_count": len(self._logged_videos)}, step=self._update_i, commit=True)
        print(f"[INFO] uploaded {uploaded} new video(s) to W&B")

    def after_update(self, metrics: dict):
        self._update_i += 1
        if self._update_i % self.upload_every == 0:
            print(f"[INFO] update {self._update_i}: attempting video upload…")
            if wandb.run is None:
                self._fatal("W&B run is not initialized in this process.")
            self._log_new_videos_or_exit()

        # Optional scalar logging
        try:
            extras = self.env.unwrapped.get_attr("extras")[0]
            for k, v in extras.items():
                if k.startswith("Rewards/") or k == "is_success":
                    wandb.log({f"train/{k}": v}, step=self._update_i, commit=False)
        except Exception:
            pass
