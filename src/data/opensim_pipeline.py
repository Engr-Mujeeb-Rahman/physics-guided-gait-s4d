"""OpenSim Muscle Analysis pipeline.

Extracts musculotendon length (l_m) and velocity (v_m) from kinematics using
the Rajagopal musculoskeletal model in OpenSim.
"""

import os
import io
import zipfile
import argparse
import numpy as np
import scipy.io

# Suppress duplicate OpenMP runtime error on Windows if PyTorch and OpenSim both loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import opensim
except ImportError:
    opensim = None


def get_default_model(model_path="data/opensim_processed/models/Rajagopal2016.osim"):
    """Load and initialize OpenSim Rajagopal model."""
    if opensim is None:
        raise ImportError("OpenSim Python API is not available in the current environment.")
    
    abs_path = os.path.abspath(model_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"OpenSim model file not found at: {abs_path}")
    
    model = opensim.Model(abs_path)
    state = model.initSystem()
    return model, state


def extract_trial_kinematics(mat_source, leg="RIGHT"):
    """Extract joint angles from MATLAB metrics_normative data.
    
    Args:
        mat_source: File path (str) or file-like bytes (io.BytesIO).
        leg: 'RIGHT' or 'LEFT'
    
    Returns:
        dict of coordinate name -> 1D numpy array of angles (radians, 100 points)
    """
    if isinstance(mat_source, (str, bytes, io.BytesIO)):
        mat_data = scipy.io.loadmat(mat_source)
    else:
        raise TypeError("mat_source must be a file path, bytes, or BytesIO object")
    
    leg_data = mat_data["metrics_normative"][leg][0, 0]
    
    # Extract kinematic curves (1x100 degrees -> radians)
    hip_flex = np.radians(leg_data["HIP"][0, 0]["FLEXION"][0, 0]["mean"][0, 0].flatten())
    hip_add = np.radians(leg_data["HIP"][0, 0]["ADDUCTION"][0, 0]["mean"][0, 0].flatten())
    hip_rot = np.radians(leg_data["HIP"][0, 0]["ROTATION"][0, 0]["mean"][0, 0].flatten())
    knee_flex = np.radians(leg_data["KNEE"][0, 0]["FLEXION"][0, 0]["mean"][0, 0].flatten())
    
    # In Rajagopal model: ankle dorsiflexion is positive, plantarflexion is negative
    ankle_pf = -np.radians(leg_data["ANKLE"][0, 0]["PLANTARFLEXION"][0, 0]["mean"][0, 0].flatten())
    ankle_inv = np.radians(leg_data["ANKLE"][0, 0]["INVERSION"][0, 0]["mean"][0, 0].flatten())
    
    pelvis_tilt = np.radians(leg_data["PELVIS"][0, 0]["TILT"][0, 0]["mean"][0, 0].flatten())
    pelvis_obl = np.radians(leg_data["PELVIS"][0, 0]["OBLIQUITY"][0, 0]["mean"][0, 0].flatten())
    pelvis_rot = np.radians(leg_data["PELVIS"][0, 0]["ROTATION"][0, 0]["mean"][0, 0].flatten())
    
    suffix = "_r" if leg == "RIGHT" else "_l"
    coord_dict = {
        "pelvis_tilt": pelvis_tilt,
        "pelvis_list": pelvis_obl,
        "pelvis_rotation": pelvis_rot,
        f"hip_flexion{suffix}": hip_flex,
        f"hip_adduction{suffix}": hip_add,
        f"hip_rotation{suffix}": hip_rot,
        f"knee_angle{suffix}": knee_flex,
        f"ankle_angle{suffix}": ankle_pf,
        f"subtalar_angle{suffix}": ankle_inv,
    }
    return coord_dict


def compute_muscle_states(model, state, coord_dict, gait_duration=1.25, leg="RIGHT"):
    """Compute musculotendon lengths (l_m) and lengthening velocities (v_m).
    
    Args:
        model: opensim.Model instance
        state: opensim.State instance
        coord_dict: dict of coordinate name -> array of angles (rad)
        gait_duration: float, estimated duration of gait cycle in seconds
        leg: 'RIGHT' or 'LEFT'
        
    Returns:
        muscle_names: list of str
        l_m: numpy array shape (n_frames, n_muscles) in meters
        v_m: numpy array shape (n_frames, n_muscles) in m/s
    """
    first_coord = next(iter(coord_dict.values()))
    n_frames = len(first_coord)
    dt = gait_duration / n_frames
    
    # Compute coordinate speeds via central differences
    coord_speeds = {name: np.gradient(vals, dt) for name, vals in coord_dict.items()}
    
    suffix = "_r" if leg == "RIGHT" else "_l"
    all_muscles = [model.getMuscles().get(i).getName() for i in range(model.getMuscles().getSize())]
    active_muscles = [m for m in all_muscles if m.endswith(suffix)]
    
    n_muscles = len(active_muscles)
    l_m = np.zeros((n_frames, n_muscles), dtype=np.float32)
    v_m = np.zeros((n_frames, n_muscles), dtype=np.float32)
    
    coord_set = model.getCoordinateSet()
    coord_objs = {}
    for name in coord_dict.keys():
        if coord_set.contains(name):
            coord_objs[name] = coord_set.get(name)
    
    muscle_objs = [opensim.Muscle.safeDownCast(model.getMuscles().get(m)) for m in active_muscles]
    
    for t in range(n_frames):
        for name, c_obj in coord_objs.items():
            c_obj.setValue(state, float(coord_dict[name][t]), False)
            c_obj.setSpeedValue(state, float(coord_speeds[name][t]))
        
        model.realizeVelocity(state)
        
        for idx, m_obj in enumerate(muscle_objs):
            l_m[t, idx] = m_obj.getLength(state)
            v_m[t, idx] = m_obj.getLengtheningSpeed(state)
            
    return active_muscles, l_m, v_m


def run_single_trial(mat_file_path_or_bytes, model_path="data/opensim_processed/models/Rajagopal2016.osim",
                     leg="RIGHT", gait_duration=1.25):
    """Full single-trial execution."""
    model, state = get_default_model(model_path)
    coord_dict = extract_trial_kinematics(mat_file_path_or_bytes, leg=leg)
    muscle_names, l_m, v_m = compute_muscle_states(model, state, coord_dict, gait_duration=gait_duration, leg=leg)
    
    # Validation check
    has_nan = np.isnan(l_m).any() or np.isnan(v_m).any()
    has_inf = np.isinf(l_m).any() or np.isinf(v_m).any()
    min_l, max_l = float(l_m.min()), float(l_m.max())
    min_v, max_v = float(v_m.min()), float(v_m.max())
    
    is_sane = (
        not has_nan and
        not has_inf and
        0.02 <= min_l <= 0.20 and
        0.30 <= max_l <= 0.80 and
        -3.0 <= min_v <= 0.0 and
        0.0 <= max_v <= 3.0
    )
    
    results = {
        "muscle_names": np.array(muscle_names),
        "l_m": l_m,
        "v_m": v_m,
        "is_sane": is_sane,
        "stats": {
            "min_lm": min_l,
            "max_lm": max_l,
            "mean_lm": float(l_m.mean()),
            "min_vm": min_v,
            "max_vm": max_v,
            "mean_vm": float(v_m.mean()),
            "has_nan": bool(has_nan),
            "has_inf": bool(has_inf),
        }
    }
    return results


def process_all_primary_trials(zip_path="data/raw/primary_six_speed_gait_dataset.zip",
                               models_dir="data/opensim_processed/models",
                               output_dir="data/opensim_processed",
                               overwrite=True):
    """Process all norm_data.mat files in the primary dataset using per-subject scaled models."""
    import time
    from tqdm import tqdm
    
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        mat_files = sorted([f for f in z.namelist() if f.endswith("norm_data.mat")])
        print(f"Found {len(mat_files)} trials to process using per-subject scaled models.")
        
        t0 = time.perf_counter()
        success_count = 0
        skip_count = 0
        failed = []
        trial_times = []
        
        current_subj = None
        current_model = None
        current_state = None
        
        for f in tqdm(mat_files, desc="OpenSim Scaled Muscle Analysis"):
            parts = f.strip("/").split("/")
            group = parts[1]  # Old or Young
            subj = parts[2]   # e.g. 16_BC
            trial_name = parts[3].replace("norm_data.mat", "") # e.g. 0_60
            
            out_filename = f"{subj}_{trial_name}_muscles.npz"
            out_path = os.path.join(output_dir, out_filename)
            
            if os.path.exists(out_path) and not overwrite:
                skip_count += 1
                success_count += 1
                continue
                
            try:
                t_trial_start = time.perf_counter()
                
                # Load subject's scaled model if switching subjects
                if subj != current_subj:
                    subj_model_path = os.path.join(models_dir, f"{subj}_scaled.osim")
                    if not os.path.exists(subj_model_path):
                        # Fallback to generic model if scaled model not found
                        subj_model_path = os.path.join(models_dir, "Rajagopal2016.osim")
                    current_model = opensim.Model(subj_model_path)
                    current_state = current_model.initSystem()
                    current_subj = subj
                    
                mat_bytes = io.BytesIO(z.read(f))
                coord_dict = extract_trial_kinematics(mat_bytes, leg="RIGHT")
                
                # Estimate gait duration from speed
                speed_str = trial_name.replace("_", ".")
                try:
                    speed_val = float(speed_str)
                    gait_duration = max(0.8, min(1.6, 1.25 / (speed_val / 1.0)))
                except ValueError:
                    gait_duration = 1.25
                    
                muscle_names, l_m, v_m = compute_muscle_states(
                    current_model, current_state, coord_dict, gait_duration=gait_duration, leg="RIGHT"
                )
                
                has_nan = bool(np.isnan(l_m).any() or np.isnan(v_m).any())
                has_inf = bool(np.isinf(l_m).any() or np.isinf(v_m).any())
                is_sane = not has_nan and not has_inf
                
                np.savez_compressed(
                    out_path,
                    subject=subj,
                    group=group,
                    trial=trial_name,
                    speed=float(trial_name.replace("_", ".")),
                    muscle_names=np.array(muscle_names),
                    l_m=l_m,
                    v_m=v_m,
                    is_sane=is_sane
                )
                
                t_trial_end = time.perf_counter()
                trial_times.append(t_trial_end - t_trial_start)
                success_count += 1
            except Exception as e:
                failed.append((f, str(e)))
                
        elapsed = time.perf_counter() - t0
        mean_trial_time = np.mean(trial_times) if trial_times else 0.0
        median_trial_time = np.median(trial_times) if trial_times else 0.0
        
        print("\n==================================================")
        print(f"Scaled Batch Processing Complete: {success_count}/{len(mat_files)} trials.")
        print(f"Total Muscle Extraction Time: {elapsed:.2f} s")
        print(f"Mean Muscle Extraction Time:  {mean_trial_time*1000:.1f} ms / trial ({mean_trial_time:.4f} s)")
        print(f"Median Muscle Extraction Time:{median_trial_time*1000:.1f} ms / trial ({median_trial_time:.4f} s)")
        print("==================================================")
        if failed:
            print(f"Failures ({len(failed)}): {failed[:5]}")
        return success_count, len(mat_files)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenSim Muscle Analysis on Gait Kinematics")
    parser.add_argument("--all", action="store_true", help="Process all trials in primary dataset")
    parser.add_argument("--zip_path", type=str, default="data/raw/primary_six_speed_gait_dataset.zip")
    parser.add_argument("--trial_path", type=str, default="Gait Dataset/Old/16_BC/0_60norm_data.mat")
    parser.add_argument("--model_path", type=str, default="data/opensim_processed/models/Rajagopal2016.osim")
    parser.add_argument("--output_file", type=str, default="data/opensim_processed/16_BC_0_60_muscles.npz")
    args = parser.parse_args()
    
    if args.all:
        process_all_primary_trials(zip_path=args.zip_path, models_dir="data/opensim_processed/models")
    else:
        print(f"Loading trial {args.trial_path} from {args.zip_path}...")
        with zipfile.ZipFile(args.zip_path, 'r') as z:
            mat_bytes = z.read(args.trial_path)
            
        res = run_single_trial(io.BytesIO(mat_bytes), model_path=args.model_path)
        print("Execution complete. Sanity check:", res["is_sane"])
        print("Stats:", res["stats"])
        
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        np.savez_compressed(
            args.output_file,
            muscle_names=res["muscle_names"],
            l_m=res["l_m"],
            v_m=res["v_m"],
            is_sane=res["is_sane"]
        )
        print(f"Results saved to {args.output_file}")
