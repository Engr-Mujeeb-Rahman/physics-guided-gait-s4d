"""Dataset splitting utilities for subject-wise and leave-one-speed-out evaluation.

Governed by architecture.md §3 and design.md §6.
Guarantees strictly disjoint subject partitions to eliminate data leakage.
"""

import numpy as np

ALL_SUBJECTS = [
    "01_LLG", "02_NR", "03_RC", "04_EP", "05_FC", "06_KS", "07_LD", "08_CF",
    "09_LV", "10_CF", "11_YD", "12_LG", "13_TP", "15_CB", "16_BC", "17_TC",
    "18_HJ", "19_AO", "20_OP", "21_JMM", "22_FM", "23_RL", "24_RC", "25_MM",
    "26_EM", "27_VL", "28_AC", "29_MDV", "30_VD", "31_SZ", "32_AL", "33_MH",
    "34_AV", "35_DLG", "36_AB", "37_SD", "38_EPG", "39_XB", "40_VL", "41_MP",
    "42_CM", "43_FD", "44_PM", "45_MC"
]

ALL_SPEEDS = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def get_subject_splits(val_ratio=0.15, test_ratio=0.15, seed=42):
    """Generate reproducible, subject-wise train/val/test splits.
    
    Args:
        val_ratio: fraction of subjects for validation (default: 0.15 -> ~6 subjects)
        test_ratio: fraction of subjects for test (default: 0.15 -> ~6 subjects)
        seed: random seed for reproducibility
        
    Returns:
        dict: {'train': list of str, 'val': list of str, 'test': list of str}
    """
    rng = np.random.RandomState(seed)
    subjects = np.array(sorted(ALL_SUBJECTS))
    shuffled = rng.permutation(subjects)
    
    n_total = len(shuffled)
    n_test = int(round(n_total * test_ratio))
    n_val = int(round(n_total * val_ratio))
    n_train = n_total - n_test - n_val
    
    train_subs = sorted(shuffled[:n_train].tolist())
    val_subs = sorted(shuffled[n_train:n_train + n_val].tolist())
    test_subs = sorted(shuffled[n_train + n_val:].tolist())
    
    # Assert mutually exclusive
    assert set(train_subs).isdisjoint(set(val_subs))
    assert set(train_subs).isdisjoint(set(test_subs))
    assert set(val_subs).isdisjoint(set(test_subs))
    assert len(train_subs) + len(val_subs) + len(test_subs) == n_total
    
    return {
        "train": train_subs,
        "val": val_subs,
        "test": test_subs
    }


def get_leave_one_speed_splits(held_out_speed, val_speed=None, seed=42):
    """Generate leave-one-speed-out split configuration.
    
    For OOD evaluation (PRD Success Criterion #2):
    The model is trained on speeds != held_out_speed (and != val_speed),
    validated on val_speed or a subset of training speeds,
    and evaluated on the held-out speed.
    
    Args:
        held_out_speed: float, e.g. 0.6 or 1.6
        val_speed: optional float, speed to use for validation. If None, uses a speed
                   adjacent to training or standard subject-split validation.
        seed: random seed
        
    Returns:
        dict with speed criteria and subject splits.
    """
    if held_out_speed not in ALL_SPEEDS:
        raise ValueError(f"held_out_speed {held_out_speed} not in {ALL_SPEEDS}")
        
    train_speeds = [s for s in ALL_SPEEDS if s != held_out_speed]
    
    subj_splits = get_subject_splits(val_ratio=0.15, test_ratio=0.15, seed=seed)
    
    return {
        "held_out_speed": held_out_speed,
        "train_speeds": train_speeds,
        "test_speed": held_out_speed,
        "subject_splits": subj_splits
    }


if __name__ == "__main__":
    splits = get_subject_splits(seed=42)
    print("Subject splits (seed=42):")
    print(f"  Train ({len(splits['train'])}): {splits['train']}")
    print(f"  Val   ({len(splits['val'])}): {splits['val']}")
    print(f"  Test  ({len(splits['test'])}): {splits['test']}")
    
    loso = get_leave_one_speed_splits(0.6)
    print("\nLeave-One-Speed-Out (0.6 m/s):")
    print(f"  Train speeds: {loso['train_speeds']}")
    print(f"  Held-out speed: {loso['held_out_speed']}")
