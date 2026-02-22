"""
Directory-based VLA dataset loader.
Loads episodes from directory structure like:
  mango/
    episode_0/
      camera1/
      ee_poses.npy
      gripper_pos.npy
      instruction.txt
      instruction_embedding.pt
"""
import os
import json
import yaml
import cv2
import numpy as np
import torch
from pathlib import Path
import re

from configs.state_vec import STATE_VEC_IDX_MAPPING
from docs.test_6drot import convert_quaternion_to_orthod6d


def natural_sort_filenames(file_list):
    """Sort filenames naturally"""
    def extract_number(filename):
        match = re.search(r'episode_(\d+)', filename)
        if match:
            return int(match.group(1))
        return 0
    return sorted(file_list, key=extract_number)


def pad_and_resize_for_siglip(images, target_size=384):
    """Pad images to square and resize to target size"""
    batch_size, height, width, channels = images.shape
    processed_images = np.zeros((batch_size, target_size, target_size, channels), dtype=images.dtype)

    for i in range(batch_size):
        img = images[i]
        max_dim = max(height, width)
        square_img = np.zeros((max_dim, max_dim, channels), dtype=img.dtype)
        
        pad_height = (max_dim - height) // 2
        pad_width = (max_dim - width) // 2
        square_img[pad_height:pad_height + height, pad_width:pad_width + width, :] = img
        
        resized_img = cv2.resize(square_img, (target_size, target_size), interpolation=cv2.INTER_AREA)
        processed_images[i] = resized_img

    return processed_images


def converted_ee_pose_with_gripper(ee_poses, gripper_pos):
    """
    Convert end-effector pose with gripper.
    
    Args:
        ee_poses: (T, 7) array of [x, y, z, qx, qy, qz, qw]
        gripper_pos: (T,) array of gripper positions
        
    Returns:
        qpos: (T, 10) array of [x, y, z, 6d_ori (6 dims), gripper]
    """
    ee_pos = ee_poses[:, :3]
    ee_ori = ee_poses[:, 3:]
    ee_6d = convert_quaternion_to_orthod6d(ee_ori)
    
    grip_pos = gripper_pos.reshape(-1, 1)
    
    qpos = np.concatenate((ee_pos, ee_6d, grip_pos), axis=-1)
    
    return qpos


class DirectoryVLADataset:
    """
    Dataset loader for directory-based episode storage.
    """
    
    def __init__(self, dataset_name=None, base_dir="data/datasets") -> None:
        if dataset_name is None:
            dataset_name = "mango"
            cfg_path = Path("configs/finetune_datasets.json")
            if cfg_path.exists():
                with open(cfg_path, "r") as f:
                    cfg_names = json.load(f)
                if isinstance(cfg_names, list) and cfg_names:
                    dataset_name = cfg_names[0]
        elif isinstance(dataset_name, (list, tuple)) and dataset_name:
            dataset_name = dataset_name[0]

        self.DATASET_NAME = dataset_name
        self.base_dir = base_dir
        self.dataset_dir = Path(base_dir) / dataset_name
        
        # Find all episode directories
        self.episode_dirs = natural_sort_filenames([
            str(d) for d in self.dataset_dir.iterdir() 
            if d.is_dir() and d.name.startswith('episode_')
        ])
        
        # Load the config
        with open('configs/base.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.CHUNK_SIZE = config['common']['action_chunk_size']
        self.IMG_HISORY_SIZE = config['common']['img_history_size']
        self.STATE_DIM = config['common']['state_dim']
        
        # Get each episode's length
        episode_lens = []
        for episode_dir in self.episode_dirs:
            _, epi_len = self.parse_episode_state_only(episode_dir)
            if epi_len is not None:
                episode_lens.append(epi_len)
        
        self.total_episode_lengths = np.sum(episode_lens)
        self.episode_sample_weights = np.array(episode_lens) / np.sum(episode_lens)
    
    def __len__(self):
        return len(self.episode_sample_weights)
    
    def get_totol_episode_lengths(self):
        return self.total_episode_lengths
    
    def get_dataset_name(self):
        return self.DATASET_NAME
    
    def get_item(self, index: int = None, state_only=False):
        """Get a training sample at a random timestep."""
        while True:
            if index is None:
                episode_dir = np.random.choice(self.episode_dirs, p=self.episode_sample_weights)
            else:
                episode_dir = self.episode_dirs[index]
            
            sample, _ = self.parse_episode(episode_dir) \
                if not state_only else self.parse_episode_state_only(episode_dir)
            
            if sample:
                return sample
            else:
                index = np.random.randint(0, len(self.episode_dirs))
    
    def parse_episode(self, episode_dir):
        """Parse an episode directory to generate a training sample."""
        episode_path = Path(episode_dir)
        
        # Load episode data
        ee_poses = np.load(episode_path / 'ee_poses.npy')
        gripper_pos = np.load(episode_path / 'gripper_pos.npy')
        
        # Load instruction embedding
        instruction_embedding_path = episode_path / 'instruction_embedding.pt'
        if instruction_embedding_path.exists():
            instruction_embedding = torch.load(instruction_embedding_path).squeeze().float().numpy()
        else:
            instruction_embedding = np.zeros(768)
        
        # Convert to qpos format
        qpos = converted_ee_pose_with_gripper(ee_poses, gripper_pos)
        num_steps = qpos.shape[0]
        
        # Drop too-short episodes
        if num_steps < 32:
            return None, None
        
        # Skip the first few still steps
        EPS = 1e-2
        qpos_delta = np.abs(qpos - qpos[0:1])
        indices = np.where(np.any(qpos_delta > EPS, axis=1))[0]
        if len(indices) > 0:
            first_idx = indices[0]
        else:
            print(f"Found no qpos that exceeds the threshold in {episode_dir}")
            return None, None
        
        # Randomly sample a timestep
        step_id = np.random.randint(first_idx - 1, num_steps - int(self.CHUNK_SIZE / 2))
        action_id = step_id + 2
        
        # Assemble metadata
        meta = {
            "dataset_name": self.DATASET_NAME,
            "#steps": num_steps,
            "step_id": step_id,
            "instruction_embedding": instruction_embedding
        }
        
        # Rescale gripper to [0, 1]
        qpos = qpos / np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 255]])
        
        target_qpos = qpos[action_id:action_id + self.CHUNK_SIZE]
        
        # Parse state and action
        state = qpos[step_id:step_id + 1]
        state_std = np.std(qpos, axis=0)
        state_mean = np.mean(qpos, axis=0)
        state_norm = np.sqrt(np.mean(qpos ** 2, axis=0))
        actions = target_qpos
        
        if actions.shape[0] < self.CHUNK_SIZE:
            # Pad actions using the last action
            actions = np.concatenate([
                actions,
                np.tile(actions[-1:], (self.CHUNK_SIZE - actions.shape[0], 1))
            ], axis=0)
        
        state = self.fill_in_state(state)
        state_indicator = self.fill_in_state(np.ones_like(state_std))
        state_std = self.fill_in_state(state_std)
        state_mean = self.fill_in_state(state_mean)
        state_norm = self.fill_in_state(state_norm)
        actions = self.fill_in_state(actions)
        
        # Load images
        camera1_dir = episode_path / 'camera1'
        camera2_dir = episode_path / 'camera2'
        
        # Get image file paths
        camera1_files = sorted(list(camera1_dir.glob("*.png")) + list(camera1_dir.glob("*.jpg")))
        camera2_files = sorted(list(camera2_dir.glob("*.png")) + list(camera2_dir.glob("*.jpg")))
        
        # Load images for the current timestep
        img_indices = list(range(max(0, step_id - self.IMG_HISORY_SIZE + 1), step_id + 1))
        img_indices = [min(i, len(camera1_files) - 1) for i in img_indices]
        
        cam_high_images = []
        cam_wrist_images = []
        
        for img_idx in img_indices:
            # Load camera1 (high)
            if img_idx < len(camera1_files):
                img = cv2.imread(str(camera1_files[img_idx]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam_high_images.append(img)
            
            # Load camera2 (wrist)
            if img_idx < len(camera2_files):
                img = cv2.imread(str(camera2_files[img_idx]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                cam_wrist_images.append(img)
        
        # Pad if necessary
        while len(cam_high_images) < self.IMG_HISORY_SIZE:
            cam_high_images.insert(0, np.zeros_like(cam_high_images[0]) if cam_high_images else np.zeros((480, 640, 3), dtype=np.uint8))
        
        while len(cam_wrist_images) < self.IMG_HISORY_SIZE:
            cam_wrist_images.insert(0, np.zeros_like(cam_wrist_images[0]) if cam_wrist_images else np.zeros((480, 640, 3), dtype=np.uint8))
        
        cam_high_images = np.array(cam_high_images)
        cam_wrist_images = np.array(cam_wrist_images)
        
        # Process images
        cam_high_images = pad_and_resize_for_siglip(cam_high_images)
        cam_wrist_images = pad_and_resize_for_siglip(cam_wrist_images)
        
        sample = {
            "state": state,
            "actions": actions,
            "state_indicator": state_indicator,
            "state_std": state_std,
            "state_mean": state_mean,
            "state_norm": state_norm,
            "cam_high": cam_high_images,
            "cam_high_mask": np.ones((self.IMG_HISORY_SIZE,), dtype=np.bool_),
            "cam_left_wrist": np.zeros((self.IMG_HISORY_SIZE, 384, 384, 3), dtype=np.uint8),
            "cam_left_wrist_mask": np.zeros((self.IMG_HISORY_SIZE,), dtype=np.bool_),
            "cam_right_wrist": cam_wrist_images,
            "cam_right_wrist_mask": np.ones((self.IMG_HISORY_SIZE,), dtype=np.bool_),
            "meta": meta
        }
        
        return sample, num_steps
    
    def parse_episode_state_only(self, episode_dir):
        """Parse episode to get only the length (for weight calculation)."""
        episode_path = Path(episode_dir)
        
        try:
            ee_poses = np.load(episode_path / 'ee_poses.npy')
            return None, ee_poses.shape[0]
        except Exception as e:
            print(f"Error loading {episode_dir}: {e}")
            return None, None
    
    def fill_in_state(self, state):
        """Fill in state vector to match expected dimensions."""
        if state.shape[-1] < self.STATE_DIM:
            padding = np.zeros((*state.shape[:-1], self.STATE_DIM - state.shape[-1]))
            state = np.concatenate([state, padding], axis=-1)
        return state
