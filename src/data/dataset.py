"""PyTorch Dataset, DataLoader, windowing, and normalization for gait kinetics.

Governed by architecture.md §3 and §4.
Merges kinematics + GRF + joint moments + l_m/v_m + speed scalar into
windowed, normalized tensors for S4D training and evaluation.
"""

import os
import sys
import io
import json
import zipfile
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader

# Ensure repository root is on sys.path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.data.splits import get_subject_splits, get_leave_one_speed_splits


class GaitDataset(Dataset):
    """PyTorch Dataset for human gait kinetics."""
    
    def __init__(self, samples, scaler=None, fit_scaler=False, include_muscles=True):
        """
        Args:
            samples: list of dicts with raw trial data
            scaler: dict with normalization statistics (x_mean, x_std, y_mean, y_std)
            fit_scaler: if True, computes scaler on these samples
            include_muscles: bool, whether to include l_m and v_m channels
        """
        self.samples = samples
        self.include_muscles = include_muscles
        
        # Build raw feature arrays
        self.X_list = []
        self.Y_list = []
        self.P_list = []
        self.Qdot_list = []
        self.Speed_list = []
        self.Subj_list = []
        self.SpeedRaw_list = []
        self.DT_list = []
        self.Mass_list = []
        
        for s in samples:
            angles = s["angles"]                          # (100, 9) in degrees
            angles_rad = np.radians(angles)               # (100, 9) in radians
            dt = s["dt"]                                  # real physical dt = stride_duration / 99
            q_dot = np.gradient(angles_rad, dt, axis=0)   # (100, 9) in rad/s
            grf = s["grf"]                                # (100, 3)
            speed_val = s["speed"]                        # float
            speed_channel = np.full((len(angles), 1), speed_val, dtype=np.float32)
            
            features = [angles, q_dot, grf]
            if self.include_muscles and "l_m" in s and "v_m" in s:
                features.extend([s["l_m"], s["v_m"]])
            features.append(speed_channel)
            
            x_arr = np.concatenate(features, axis=1).astype(np.float32) # (100, Din)
            y_arr = s["moments"].astype(np.float32)                      # (100, 3) in N*m/kg
            p_arr = s["powers"].astype(np.float32)                       # (100, 3) in W/kg
            
            # Extract sagittal plane angular velocities (hip flex, knee flex, ankle plantar) in rad/s
            # In angles: index 3 is hip_flexion, index 6 is knee_flexion, index 7 is ankle_plantarflexion
            sagittal_qdot = q_dot[:, [3, 6, 7]].astype(np.float32)       # (100, 3) in rad/s
            
            self.X_list.append(x_arr)
            self.Y_list.append(y_arr)
            self.P_list.append(p_arr)
            self.Qdot_list.append(sagittal_qdot)
            self.Speed_list.append(np.float32(speed_val))
            self.Subj_list.append(s["subject"])
            self.SpeedRaw_list.append(speed_val)
            self.DT_list.append(np.float32(dt))
            self.Mass_list.append(np.float32(s.get("mass", 75.0)))
            
        # Fit or apply normalization
        if fit_scaler:
            all_x = np.concatenate(self.X_list, axis=0) # (N*100, Din)
            all_y = np.concatenate(self.Y_list, axis=0) # (N*100, 3)
            
            x_mean = all_x.mean(axis=0, keepdims=True)
            x_std = all_x.std(axis=0, keepdims=True)
            x_std[x_std < 1e-6] = 1.0 # prevent div-by-zero
            
            y_mean = all_y.mean(axis=0, keepdims=True)
            y_std = all_y.std(axis=0, keepdims=True)
            y_std[y_std < 1e-6] = 1.0
            
            # Speed min/max for FiLM conditioning normalization in [0, 1]
            speeds = np.array(self.Speed_list)
            speed_min = float(speeds.min())
            speed_max = float(speeds.max())
            if speed_max == speed_min:
                speed_max = speed_min + 1.0
                
            self.scaler = {
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
                "speed_min": speed_min,
                "speed_max": speed_max
            }
        else:
            self.scaler = scaler
            
    def __len__(self):
        return len(self.X_list)
        
    def __getitem__(self, idx):
        x = self.X_list[idx]
        y = self.Y_list[idx]
        p = self.P_list[idx]
        q_dot = self.Qdot_list[idx]
        speed = self.Speed_list[idx]
        dt = self.DT_list[idx]
        mass = self.Mass_list[idx]
        
        # Apply normalization if scaler provided
        if self.scaler is not None:
            x = (x - self.scaler["x_mean"]) / self.scaler["x_std"]
            speed_norm = (speed - self.scaler["speed_min"]) / (self.scaler["speed_max"] - self.scaler["speed_min"])
        else:
            speed_norm = speed
            
        return {
            "x": torch.from_numpy(x).float(),                         # (100, Din)
            "y": torch.from_numpy(y).float(),                         # (100, 3)
            "speed": torch.tensor([speed_norm], dtype=torch.float32), # (1,)
            "angular_vel": torch.from_numpy(q_dot).float(),           # (100, 3) in rad/s
            "power_meas": torch.from_numpy(p).float(),                # (100, 3) in W/kg
            "dt": torch.tensor([dt], dtype=torch.float32),            # (1,) in s
            "mass": torch.tensor([mass], dtype=torch.float32),        # (1,) in kg
            "subject": self.Subj_list[idx],
            "speed_raw": self.SpeedRaw_list[idx]
        }


def load_all_primary_samples(zip_path="data/raw/primary_six_speed_gait_dataset.zip",
                             opensim_dir="data/opensim_processed"):
    """Load and index all trials from primary dataset zip and opensim_processed."""
    samples = []
    
    # Load per-trial stride durations if available
    dur_path = os.path.join(opensim_dir, "trial_stride_durations.json")
    if os.path.exists(dur_path):
        with open(dur_path, "r") as f_dur:
            stride_durations = json.load(f_dur)
    else:
        stride_durations = {}
        
    # Load per-subject dynamic GRF masses if available
    mass_path = os.path.join(opensim_dir, "subject_masses.json")
    if os.path.exists(mass_path):
        with open(mass_path, "r") as f_mass:
            subject_masses = json.load(f_mass)
    else:
        subject_masses = {}
    
    with zipfile.ZipFile(zip_path, "r") as z:
        mat_files = [f for f in z.namelist() if f.endswith("norm_data.mat")]
        
        for f in mat_files:
            parts = f.strip("/").split("/")
            group = parts[1]
            subj = parts[2]
            trial_name = parts[3].replace("norm_data.mat", "") # e.g. 0_60
            
            speed_str = trial_name.replace("_", ".")
            try:
                speed_val = float(speed_str)
            except ValueError:
                speed_val = 1.0
                
            # Real physical stride duration and dt
            trial_key = f"{subj}_{trial_name}"
            if trial_key in stride_durations:
                gait_duration = stride_durations[trial_key]["stride_duration_seconds"]
                dt_val = stride_durations[trial_key]["dt_seconds"]
            else:
                gait_duration = max(0.8, min(1.6, 1.25 / (speed_val / 1.0)))
                dt_val = gait_duration / 99.0
                
            # Dynamic GRF body mass (kg)
            subj_mass = subject_masses.get(subj, {}).get("mass_kg", 75.0)
                
            # Read MATLAB kinetics and kinematics
            mat = scipy.io.loadmat(io.BytesIO(z.read(f)))["metrics_normative"]["RIGHT"][0, 0]
            
            angles = np.stack([
                mat["PELVIS"][0,0]["TILT"][0,0]["mean"][0,0].flatten(),
                mat["PELVIS"][0,0]["OBLIQUITY"][0,0]["mean"][0,0].flatten(),
                mat["PELVIS"][0,0]["ROTATION"][0,0]["mean"][0,0].flatten(),
                mat["HIP"][0,0]["FLEXION"][0,0]["mean"][0,0].flatten(),
                mat["HIP"][0,0]["ADDUCTION"][0,0]["mean"][0,0].flatten(),
                mat["HIP"][0,0]["ROTATION"][0,0]["mean"][0,0].flatten(),
                mat["KNEE"][0,0]["FLEXION"][0,0]["mean"][0,0].flatten(),
                mat["ANKLE"][0,0]["PLANTARFLEXION"][0,0]["mean"][0,0].flatten(),
                mat["ANKLE"][0,0]["INVERSION"][0,0]["mean"][0,0].flatten()
            ], axis=1) # (100, 9) in degrees
            
            grf = np.stack([
                mat["FORCE"][0,0]["Fx"][0,0]["mean"][0,0].flatten(),
                mat["FORCE"][0,0]["Fy"][0,0]["mean"][0,0].flatten(),
                mat["FORCE"][0,0]["Fz"][0,0]["mean"][0,0].flatten()
            ], axis=1) # (100, 3)
            
            moments = np.stack([
                mat["MOMENT_ARTICULAIRE"][0,0]["M_hip"][0,0]["mean"][0,0].flatten(),
                mat["MOMENT_ARTICULAIRE"][0,0]["M_knee"][0,0]["mean"][0,0].flatten(),
                mat["MOMENT_ARTICULAIRE"][0,0]["M_ankle"][0,0]["mean"][0,0].flatten()
            ], axis=1) # (100, 3) in N*m/kg
            
            powers = np.stack([
                mat["PUISSANCE_ARTICULAIRE"][0,0]["P_hip"][0,0]["mean"][0,0].flatten(),
                mat["PUISSANCE_ARTICULAIRE"][0,0]["P_knee"][0,0]["mean"][0,0].flatten(),
                mat["PUISSANCE_ARTICULAIRE"][0,0]["P_ankle"][0,0]["mean"][0,0].flatten()
            ], axis=1) # (100, 3) in W/kg
            
            # Read OpenSim processed muscle states
            muscle_npz = os.path.join(opensim_dir, f"{subj}_{trial_name}_muscles.npz")
            if os.path.exists(muscle_npz):
                m_data = np.load(muscle_npz)
                l_m = m_data["l_m"] # (100, 40)
                v_m = m_data["v_m"] # (100, 40)
            else:
                l_m = None
                v_m = None
                
            sample = {
                "subject": subj,
                "group": group,
                "trial": trial_name,
                "speed": speed_val,
                "gait_duration": gait_duration,
                "dt": dt_val,
                "mass": subj_mass,
                "angles": angles,
                "grf": grf,
                "moments": moments,
                "powers": powers,
                "l_m": l_m,
                "v_m": v_m
            }
            samples.append(sample)
            
    return samples


def get_dataloaders(batch_size=16, split_mode="subject", held_out_speed=None,
                    include_muscles=True, seed=42, num_workers=0):
    """Construct PyTorch DataLoaders for train, val, and test splits.
    
    Args:
        batch_size: int (default: 16)
        split_mode: 'subject' (standard subject split) or 'loso' (leave-one-speed-out)
        held_out_speed: float, required if split_mode == 'loso'
        include_muscles: bool, whether to include muscle channels
        seed: random seed for reproducibility
        num_workers: DataLoader num_workers (0 on Windows)
        
    Returns:
        train_loader, val_loader, test_loader, scaler
    """
    all_samples = load_all_primary_samples()
    
    if split_mode == "subject":
        splits = get_subject_splits(seed=seed)
        train_subs = set(splits["train"])
        val_subs = set(splits["val"])
        test_subs = set(splits["test"])
        
        train_samples = [s for s in all_samples if s["subject"] in train_subs]
        val_samples = [s for s in all_samples if s["subject"] in val_subs]
        test_samples = [s for s in all_samples if s["subject"] in test_subs]
        
    elif split_mode == "loso":
        if held_out_speed is None:
            raise ValueError("held_out_speed must be provided for loso split")
            
        splits = get_leave_one_speed_splits(held_out_speed, seed=seed)
        train_subs = set(splits["subject_splits"]["train"])
        val_subs = set(splits["subject_splits"]["val"])
        test_subs = set(splits["subject_splits"]["test"])
        
        # Train on train subjects and speeds != held_out_speed
        train_samples = [
            s for s in all_samples 
            if s["subject"] in train_subs and abs(s["speed"] - held_out_speed) > 1e-3
        ]
        val_samples = [
            s for s in all_samples 
            if s["subject"] in val_subs and abs(s["speed"] - held_out_speed) > 1e-3
        ]
        # Test on the held-out speed across test subjects (or held-out speed across all subjects)
        test_samples = [
            s for s in all_samples 
            if abs(s["speed"] - held_out_speed) < 1e-3
        ]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
        
    # Fit scaler on train split only!
    train_dataset = GaitDataset(train_samples, fit_scaler=True, include_muscles=include_muscles)
    scaler = train_dataset.scaler
    
    val_dataset = GaitDataset(val_samples, scaler=scaler, fit_scaler=False, include_muscles=include_muscles)
    test_dataset = GaitDataset(test_samples, scaler=scaler, fit_scaler=False, include_muscles=include_muscles)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, scaler


if __name__ == "__main__":
    train_loader, val_loader, test_loader, scaler = get_dataloaders(batch_size=16, include_muscles=True)
    print(f"Train batches: {len(train_loader)} (total samples: {len(train_loader.dataset)})")
    print(f"Val batches:   {len(val_loader)} (total samples: {len(val_loader.dataset)})")
    print(f"Test batches:  {len(test_loader)} (total samples: {len(test_loader.dataset)})")
    
    batch = next(iter(train_loader))
    print("\n--- Batch Inspection ---")
    print("x shape:          ", batch["x"].shape)
    print("y shape:          ", batch["y"].shape)
    print("speed shape:      ", batch["speed"].shape)
    print("angular_vel shape:", batch["angular_vel"].shape)
    print("power_meas shape: ", batch["power_meas"].shape)
    print("x mean / std:     ", batch["x"].mean().item(), batch["x"].std().item())
    print("y min / max:      ", batch["y"].min().item(), batch["y"].max().item())
