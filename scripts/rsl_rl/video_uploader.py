import glob
import os
import wandb


def upload_videos_to_wandb(cfg):
    # find all train videos and log them
    pattern = os.path.join(cfg.runner.log_dir, "videos", "train", "**", "*.mp4")
    for path in glob.glob(pattern, recursive=True):
        wandb.log({"train/policy_video": wandb.Video(path, fps=30)}, commit=False)
    wandb.log({}, commit=True)


def upload_video_file(path):
    # log a single eval video
    if os.path.exists(path):
        wandb.log({"eval/policy_video": wandb.Video(path, fps=30)})
