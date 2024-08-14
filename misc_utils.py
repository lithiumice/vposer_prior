import torch
import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt
from glob import glob


def get_random_files(directory: Path, num_files: int) -> list:
    """Get a random sample of npz files from the specified directory."""
    all_files = list(directory.glob('*.npz'))
    return random.sample(all_files, num_files)

def extract_poses(file_path: Path) -> torch.Tensor:
    """Extract body poses from an npz file."""
    amass_body_pose = np.load(file_path)['poses'][:, 1:22].reshape(-1, 63)
    return torch.from_numpy(amass_body_pose).type(torch.float)

