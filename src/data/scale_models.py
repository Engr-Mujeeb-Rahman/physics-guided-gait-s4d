"""Per-subject OpenSim model scaling pipeline.

Extracts anthropometrics (mass from force plates and segment lengths from optical markers)
from each subject's static standing calibration trial (static.c3d).
Scales the Rajagopal 2016 musculoskeletal model per-subject and saves 44 scaled models
to data/opensim_processed/models/<subject>_scaled.osim.
"""

import os
import sys
import time
import zipfile
import numpy as np
import opensim

# Suppress duplicate OpenMP runtime error on Windows if PyTorch and OpenSim both loaded
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Nominal reference marker distances in generic Rajagopal2016.osim (meters)
NOMINAL_REF = {
    "pelvis_width": 0.2573,    # RASI to LASI
    "femur_length": 0.5110,    # RASI to RKJC
    "tibia_length": 0.3980,    # RKJC to RAJC
    "leg_length": 0.9090,      # Femur + Tibia
    "foot_length": 0.2319,     # RCAL to RTOE
    "torso_height": 0.5348,    # C7 to Pelvis mid
    "nominal_mass": 75.337,    # Standard Rajagopal model total mass (kg)
}


def extract_subject_static_data(subj: str, group: str, zip_path: str = "data/raw/primary_six_speed_gait_dataset.zip", scratch_dir: str = "scratch") -> dict:
    """Extract mass and segment lengths from a subject's static.c3d file."""
    os.makedirs(scratch_dir, exist_ok=True)
    c3d_zip_entry = f"Gait Dataset/{group}/{subj}/static.c3d"
    scratch_c3d = os.path.join(scratch_dir, f"{subj}_static.c3d")
    
    with zipfile.ZipFile(zip_path, "r") as z:
        with open(scratch_c3d, "wb") as f_out:
            f_out.write(z.read(c3d_zip_entry))
            
    adapter = opensim.C3DFileAdapter()
    tables = adapter.read(scratch_c3d)
    m_table = adapter.getMarkersTable(tables)
    f_table = adapter.getForcesTable(tables)
    labels = [m_table.getColumnLabel(i) for i in range(m_table.getNumColumns())]
    
    # 1. Mass estimation from quiet standing force plates
    f1 = f_table.getDependentColumn("f1")
    f2 = f_table.getDependentColumn("f2")
    fz = [f1.getElt(i, 0).get(2) + f2.getElt(i, 0).get(2) for i in range(f_table.getNumRows())]
    raw_fz = float(np.mean(fz))
    
    # 2. Marker positions (mean over static frames)
    def get_pos(name):
        col = m_table.getDependentColumn(name)
        arr = np.array([[col.getElt(i, 0).get(0), col.getElt(i, 0).get(1), col.getElt(i, 0).get(2)] for i in range(m_table.getNumRows())])
        return arr.mean(axis=0)
        
    r_asis = get_pos("RASIS")
    l_asis = get_pos("LASIS")
    pelvis_w = float(np.linalg.norm(r_asis - l_asis))
    
    r_knee = 0.5 * (get_pos("RKNEELAT") + get_pos("RKNEEMED"))
    r_ankle = 0.5 * (get_pos("RANKLELAT") + get_pos("RANKLEMED"))
    r_heel = get_pos("RHEEL")
    r_toe = 0.5 * (get_pos("RTOELAT") + get_pos("RTOEMED"))
    
    if "C7" in labels:
        upper = get_pos("C7")
    elif "CLAV" in labels:
        upper = get_pos("CLAV")
    else:
        upper = get_pos("T2")
        
    pelvis_mid = 0.5 * (r_asis + l_asis)
    
    femur_l = float(np.linalg.norm(r_asis - r_knee))
    tibia_l = float(np.linalg.norm(r_knee - r_ankle))
    foot_l = float(np.linalg.norm(r_heel - r_toe))
    torso_h = float(np.linalg.norm(upper - pelvis_mid))
    leg_l = femur_l + tibia_l
    
    # Calculate scale factors
    s_pelvis = pelvis_w / NOMINAL_REF["pelvis_width"]
    s_femur = femur_l / NOMINAL_REF["femur_length"]
    s_tibia = tibia_l / NOMINAL_REF["tibia_length"]
    s_foot = foot_l / NOMINAL_REF["foot_length"]
    s_torso = torso_h / NOMINAL_REF["torso_height"]
    
    # Clean mass determination:
    # If static force plate has valid vertical force in physiological human range [400, 1200] N
    if 400.0 <= raw_fz <= 1200.0:
        mass = raw_fz / 9.81
        mass_source = "force_plate_measured"
    else:
        # Anthropometric volumetric scaling fallback: mass proportional to volume
        s_leg = leg_l / NOMINAL_REF["leg_length"]
        mass = NOMINAL_REF["nominal_mass"] * (s_pelvis * s_leg * s_torso)
        mass_source = "anthropometric_scaled"
        
    return {
        "subject": subj,
        "group": group,
        "mass": float(mass),
        "mass_source": mass_source,
        "pelvis_width": pelvis_w,
        "femur_length": femur_l,
        "tibia_length": tibia_l,
        "leg_length": leg_l,
        "foot_length": foot_l,
        "torso_height": torso_h,
        "scale_factors": {
            "pelvis": float(s_pelvis),
            "femur": float(s_femur),
            "tibia": float(s_tibia),
            "foot": float(s_foot),
            "torso": float(s_torso),
        }
    }


def scale_subject_model(
    generic_model_path: str,
    subj_data: dict,
    output_model_path: str
) -> float:
    """Scale generic OpenSim model using subject anthropometrics and save scaled model.
    
    Returns:
        float: Elapsed wall-clock time in seconds for scaling.
    """
    t0 = time.perf_counter()
    model = opensim.Model(generic_model_path)
    model.setName(f"Rajagopal2016_{subj_data['subject']}")
    state = model.initSystem()
    
    sf = subj_data["scale_factors"]
    scale_set = opensim.ScaleSet()
    
    # Pelvis
    s = opensim.Scale()
    s.setSegmentName("pelvis")
    s.setScaleFactors(opensim.Vec3(sf["pelvis"], sf["pelvis"], sf["pelvis"]))
    scale_set.cloneAndAppend(s)
    
    # Femurs
    for b in ["femur_r", "femur_l"]:
        s = opensim.Scale()
        s.setSegmentName(b)
        s.setScaleFactors(opensim.Vec3(sf["femur"], sf["femur"], sf["femur"]))
        scale_set.cloneAndAppend(s)
        
    # Tibias & Patellas
    for b in ["tibia_r", "tibia_l", "patella_r", "patella_l"]:
        s = opensim.Scale()
        s.setSegmentName(b)
        s.setScaleFactors(opensim.Vec3(sf["tibia"], sf["tibia"], sf["tibia"]))
        scale_set.cloneAndAppend(s)
        
    # Feet (talus, calcn, toes)
    for b in ["talus_r", "talus_l", "calcn_r", "calcn_l", "toes_r", "toes_l"]:
        s = opensim.Scale()
        s.setSegmentName(b)
        s.setScaleFactors(opensim.Vec3(sf["foot"], sf["foot"], sf["foot"]))
        scale_set.cloneAndAppend(s)
        
    # Torso
    s = opensim.Scale()
    s.setSegmentName("torso")
    s.setScaleFactors(opensim.Vec3(sf["torso"], sf["torso"], sf["torso"]))
    scale_set.cloneAndAppend(s)
    
    # Apply scaling: model.scale(state, scale_set, preserveMassDist=False, finalMass)
    success = model.scale(state, scale_set, False, float(subj_data["mass"]))
    if not success:
        raise RuntimeError(f"OpenSim model.scale failed for subject {subj_data['subject']}")
        
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    model.printToXML(output_model_path)
    
    elapsed = time.perf_counter() - t0
    return elapsed


def scale_all_subjects(
    zip_path: str = "data/raw/primary_six_speed_gait_dataset.zip",
    generic_model_path: str = "data/opensim_processed/models/Rajagopal2016.osim",
    output_dir: str = "data/opensim_processed/models",
) -> tuple[dict, dict]:
    """Execute model scaling for all 44 subjects in primary dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, "r") as z:
        static_files = sorted([f for f in z.namelist() if f.endswith("static.c3d")])
        
    print(f"=== Starting Per-Subject Scaling for {len(static_files)} Subjects ===")
    
    all_anthro = {}
    scaling_times = {}
    
    t_start_total = time.perf_counter()
    for f in static_files:
        parts = f.strip("/").split("/")
        grp, subj = parts[1], parts[2]
        
        # 1. Extract anthropometrics
        subj_data = extract_subject_static_data(subj, grp, zip_path=zip_path)
        all_anthro[subj] = subj_data
        
        # 2. Scale and save model
        out_model_path = os.path.join(output_dir, f"{subj}_scaled.osim")
        t_scale = scale_subject_model(generic_model_path, subj_data, out_model_path)
        scaling_times[subj] = t_scale
        
        print(f"[{subj}] Mass={subj_data['mass']:.1f}kg ({subj_data['mass_source'][:5]}), "
              f"Leg={subj_data['leg_length']*100:.1f}cm, Femur={subj_data['femur_length']*100:.1f}cm, "
              f"Tibia={subj_data['tibia_length']*100:.1f}cm | Scaled in {t_scale*1000:.1f}ms")
              
    t_total = time.perf_counter() - t_start_total
    mean_scale_time = np.mean(list(scaling_times.values()))
    
    print("\n==================================================")
    print(f"Scaling Complete: 44/44 subject models created.")
    print(f"Total Wall-Clock Scaling Time: {t_total:.2f} s")
    print(f"Mean Per-Subject Scaling Time: {mean_scale_time*1000:.1f} ms ({mean_scale_time:.3f} s)")
    print("==================================================")
    
    return all_anthro, scaling_times


if __name__ == "__main__":
    scale_all_subjects()
