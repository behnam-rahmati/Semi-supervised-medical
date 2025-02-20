import shutil
from pathlib import Path
import h5py
import numpy as np
import torch
import tqdm
import tyro
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME, LeRobotDataset


def create_maniskill_dataset(repo_id: str, robot_type: str, mode="image"):
    features = {
        "observation.state": {"dtype": "float32", "shape": (7,), "names": ["state"]},
        "action": {"dtype": "float32", "shape": (7,), "names": ["action"]},
        "observation.image": {"dtype": mode, "shape": (3, 480, 640), "names": ["channels", "height", "width"]},
    }
    
    dataset_path = LEROBOT_HOME / repo_id
    if dataset_path.exists():
        shutil.rmtree(dataset_path)
    
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=30,
        robot_type=robot_type,
        features=features,
        use_videos=False,
        image_writer_threads=5,
        image_writer_processes=3,
    )


def load_maniskill_episode_data(ep_path: Path):
    with h5py.File(ep_path, "r") as ep:
        state = torch.from_numpy(ep["/observations/qpos"][:])
        action = torch.from_numpy(ep["/actions"][:])
        image = np.array(ep["/observations/rgb"][:])  # Adjust key based on ManiSkill format
    
    return state, action, image


def convert_maniskill(
    raw_dir: Path,
    repo_id: str,
    task: str,
    push_to_hub: bool = False,
):
    hdf5_files = sorted(raw_dir.glob("*.h5"))
    dataset = create_maniskill_dataset(repo_id, robot_type="sawyer")
    
    for ep_path in tqdm.tqdm(hdf5_files):
        state, action, image = load_maniskill_episode_data(ep_path)
        num_frames = state.shape[0]
        
        for i in range(num_frames):
            dataset.add_frame({
                "observation.state": state[i],
                "action": action[i],
                "observation.image": image[i],
            })
        
        dataset.save_episode(task=task)
    
    dataset.consolidate()
    
    if push_to_hub:
        dataset.push_to_hub()


if __name__ == "__main__":
    tyro.cli(convert_maniskill)
