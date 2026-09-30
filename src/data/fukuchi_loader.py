"""Fukuchi 2018 dataset loader for zero-shot cross-dataset evaluation.

Loads treadmill walking trials from Fukuchi et al. (2018, PeerJ, DOI: 10.7717/peerj.4640),
resamples them to 100 time points per gait cycle, and maps them to the identical
102-channel input feature tensor and 3-channel joint moment target tensor used
by the primary dataset model.
"""

import os
import io
import zipfile
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

# Suppress duplicate OpenMP runtime error on Windows if PyTorch and OpenSim both loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    from src.data.opensim_pipeline import get_default_model, compute_muscle_states
    HAS_OPENSIM = True
except ImportError:
    HAS_OPENSIM = False


def resample_100(arr: np.ndarray) -> np.ndarray:
    """Resample 1D or 2D array along time axis (axis 0) to exactly 100 frames."""
    n_in = arr.shape[0]
    if n_in == 100:
        return arr
    x_old = np.linspace(0, 1, n_in)
    x_new = np.linspace(0, 1, 100)
    if arr.ndim == 1:
        f = interp1d(x_old, arr, kind="cubic", fill_value="extrapolate")
        return f(x_new)
    else:
        out = np.zeros((100, arr.shape[1]), dtype=arr.dtype)
        for col in range(arr.shape[1]):
            f = interp1d(x_old, arr[:, col], kind="cubic", fill_value="extrapolate")
            out[:, col] = f(x_new)
        return out


def compute_angular_velocity(angles: np.ndarray, gait_duration: float = 1.25) -> np.ndarray:
    """Compute angular velocity (rad/s) via central differences."""
    dt = gait_duration / angles.shape[0]
    return np.gradient(angles, dt, axis=0)


def get_fukuchi_treadmill_catalog(
    fukuchi_dir: str = "data/raw/fukuchi",
) -> pd.DataFrame:
    """Load metadata catalog of all treadmill trials with valid speed from WBDSinfo.xlsx."""
    info_path = os.path.join(fukuchi_dir, "WBDSinfo.xlsx")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"WBDSinfo.xlsx not found at {info_path}")

    df_info = pd.read_excel(info_path)
    
    # Filter for treadmill trials (walkT)
    treadmill_mask = df_info["FileName"].str.contains(r"walkT\d+\.c3d", regex=True, na=False)
    df_t = df_info[treadmill_mask].copy()
    
    # Clean speed column
    df_t["speed_num"] = pd.to_numeric(df_t["GaitSpeed(m/s)"], errors="coerce")
    df_t = df_t.dropna(subset=["speed_num"])
    
    # Base trial name (e.g. WBDS01walkT01)
    df_t["TrialBase"] = df_t["FileName"].str.replace(r"\.c3d$", "", regex=True)
    return df_t


def load_fukuchi_trial_data(
    trial_base: str,
    speed: float,
    zip_path: str = "data/raw/fukuchi/WBDSascii.zip",
    compute_muscles: bool = True,
    model=None,
    state=None,
) -> dict:
    """Load and process a single Fukuchi treadmill trial.
    
    Args:
        trial_base: e.g. 'WBDS01walkT01'
        speed: Walking speed in m/s
        zip_path: Path to WBDSascii.zip
        compute_muscles: If True, computes 40 lm and 40 vm via OpenSim
        model: Pre-initialized opensim.Model
        state: Pre-initialized opensim.State
        
    Returns:
        dict with:
            'features': (100, 102) float32 array
            'targets': (100, 3) float32 array (sagittal moments: hip, knee, ankle)
            'speed': float
            'trial_name': str
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        ang_file = f"{trial_base}ang.txt"
        knt_file = f"{trial_base}knt.txt"
        
        with z.open(ang_file) as f:
            df_ang = pd.read_csv(f, sep="\t")
        with z.open(knt_file) as f:
            df_knt = pd.read_csv(f, sep="\t")

    # Joint angles (degrees -> radians) resampled to 100 points
    # In Fukuchi 2018 (PeerJ): Z is sagittal (flexion/extension), X is frontal, Y is transverse
    hip_flex = np.radians(resample_100(df_ang["RHipAngleZ"].values))
    hip_add = np.radians(resample_100(df_ang["RHipAngleX"].values))
    hip_rot = np.radians(resample_100(df_ang["RHipAngleY"].values))
    knee_flex = np.radians(resample_100(df_ang["RKneeAngleZ"].values))
    # Rajagopal ankle angle: dorsiflexion is positive; in Fukuchi ankle plantarflexion is negative
    ankle_flex = np.radians(resample_100(df_ang["RAnkleAngleZ"].values))
    subtalar_angle = np.radians(resample_100(df_ang["RAnkleAngleX"].values))
    
    pelvis_tilt = np.radians(resample_100(df_ang["RPelvisAngleZ"].values))
    pelvis_list = np.radians(resample_100(df_ang["RPelvisAngleX"].values))
    pelvis_rot = np.radians(resample_100(df_ang["RPelvisAngleY"].values))
    
    angles_9 = np.stack([
        hip_flex, hip_add, hip_rot,
        knee_flex, ankle_flex, subtalar_angle,
        pelvis_tilt, pelvis_list, pelvis_rot
    ], axis=1)  # shape (100, 9)
    
    # Angular velocities (rad/s)
    gait_duration = max(0.8, 1.25 / (speed / 1.2))  # rough stride duration scaling
    vels_9 = compute_angular_velocity(angles_9, gait_duration=gait_duration)  # (100, 9)
    
    # Ground reaction forces (normalized by body mass in N/kg)
    # In Fukuchi, swing phase (foot in air) is coded as 0.0 in early subjects and NaN in later subjects;
    # fillna(0.0) enforces physical zero GRF during swing phase across all subjects.
    grf_x = resample_100(df_knt["RGRFX"].fillna(0.0).values)
    grf_y = resample_100(df_knt["RGRFY"].fillna(0.0).values)
    grf_z = resample_100(df_knt["RGRFZ"].fillna(0.0).values)
    grf_3 = np.stack([grf_x, grf_y, grf_z], axis=1)  # (100, 3)
    
    # Muscle states
    if compute_muscles and HAS_OPENSIM:
        if model is None or state is None:
            model, state = get_default_model()
        coord_dict = {
            "hip_flexion_r": hip_flex,
            "hip_adduction_r": hip_add,
            "hip_rotation_r": hip_rot,
            "knee_angle_r": knee_flex,
            "ankle_angle_r": -ankle_flex,  # Rajagopal sign convention
            "subtalar_angle_r": subtalar_angle,
            "pelvis_tilt": pelvis_tilt,
            "pelvis_list": pelvis_list,
            "pelvis_rotation": pelvis_rot,
        }
        _, l_m, v_m = compute_muscle_states(model, state, coord_dict, gait_duration=gait_duration)
    else:
        # Placeholder zeros if OpenSim is disabled
        l_m = np.zeros((100, 40), dtype=np.float32)
        v_m = np.zeros((100, 40), dtype=np.float32)
        
    # Speed channel (scalar broadcast)
    speed_col = np.full((100, 1), speed, dtype=np.float32)
    
    # Assemble 102 channels: 9 angles + 9 vels + 3 grf + 40 lm + 40 vm + 1 speed
    features = np.concatenate([
        angles_9.astype(np.float32),
        vels_9.astype(np.float32),
        grf_3.astype(np.float32),
        l_m.astype(np.float32),
        v_m.astype(np.float32),
        speed_col
    ], axis=1)  # (100, 102)
    
    # Joint moment targets (sagittal: hip, knee, ankle in N*m/kg)
    # In Fukuchi: RHipMomentZ, RKneeMomentZ, RAnkleMomentZ
    hip_moment = resample_100(df_knt["RHipMomentZ"].values)
    knee_moment = resample_100(df_knt["RKneeMomentZ"].values)
    ankle_moment = resample_100(df_knt["RAnkleMomentZ"].values)
    targets = np.stack([hip_moment, knee_moment, ankle_moment], axis=1).astype(np.float32)  # (100, 3)
    
    return {
        "features": features,
        "targets": targets,
        "speed": float(speed),
        "trial_name": trial_base,
    }


def verify_fukuchi_compatibility():
    """Verify that Fukuchi dataset can be loaded into exact tensors matching primary dataset."""
    print("Verifying Fukuchi 2018 dataset compatibility...")
    catalog = get_fukuchi_treadmill_catalog()
    print(f"Discovered {len(catalog)} valid treadmill trials across {catalog['Subject'].nunique()} subjects.")
    
    # Test first trial
    first_row = catalog.iloc[0]
    trial_base = first_row["TrialBase"]
    speed = float(first_row["speed_num"])
    print(f"Loading sample trial: {trial_base} (speed: {speed} m/s)...")
    
    data = load_fukuchi_trial_data(trial_base, speed, compute_muscles=True)
    features = data["features"]
    targets = data["targets"]
    
    print(f"Features shape: {features.shape} (Expected: (100, 102))")
    print(f"Targets shape: {targets.shape} (Expected: (100, 3))")
    
    assert features.shape == (100, 102), f"Bad features shape: {features.shape}"
    assert targets.shape == (100, 3), f"Bad targets shape: {targets.shape}"
    assert not np.isnan(features).any(), "NaNs in features!"
    assert not np.isnan(targets).any(), "NaNs in targets!"
    
    print("CONFIRMED: Fukuchi dataset produces 100% compatible feature format for cross-dataset zero-shot evaluation.")
    return True


if __name__ == "__main__":
    verify_fukuchi_compatibility()
