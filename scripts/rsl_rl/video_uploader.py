import glob
import os
import wandb


def upload_videos_to_wandb(log_dir: str):
    """Bulk-upload all training MP4s under <log_dir>/videos/train to W&B."""
    pattern = os.path.join(log_dir, "videos", "train", "**", "*.mp4")
    for path in glob.glob(pattern, recursive=True):
        wandb.log({"train/policy_video": wandb.Video(path, fps=30)}, commit=False)
    wandb.log({}, commit=True)


def upload_video_file(path: str):
    """Upload a single eval MP4 to W&B."""
    if os.path.exists(path):
        wandb.log({"eval/policy_video": wandb.Video(path, fps=30)})
