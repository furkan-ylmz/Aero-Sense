#!/usr/bin/env python3
"""
Aero-Sense: Synthetic Kinematic Flight Trajectory Generator
Generates balanced 6-class tactical aircraft maneuver dataset with realistic 6-DOF physics,
sensor noise, feature extraction (x, y, z, v_g, theta, omega, a_t, a_z), and Z-Score normalization.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


# Maneuver Class Definitions
CLASS_NAMES = {
    0: "MANEUVER_STRAIGHT",
    1: "MANEUVER_TURN",
    2: "MANEUVER_CLIMB",
    3: "MANEUVER_DESCENT",
    4: "MANEUVER_ORBIT",
    5: "MANEUVER_EVASIVE",
}


def normalize_angle_diff_deg(delta_deg: float) -> float:
    """Normalizes angle difference to [-180.0, 180.0] degrees to prevent 360->0 boundary jumps."""
    return (delta_deg + 180.0) % 360.0 - 180.0


def generate_single_trajectory(
    maneuver_class: int,
    duration_sec: int = 40,
    dt: float = 1.0,
    noise_std: float = 0.05,
) -> Tuple[np.ndarray, int]:
    """
    Generates a single flight trajectory for a given maneuver class.
    Returns:
        features: (T, 8) numpy array containing [x, y, z, v_g, theta, omega, a_t, a_z]
        label: maneuver class ID (0 to 5)
    """
    num_steps = int(duration_sec / dt)

    # Initial state randomization
    x0 = np.random.uniform(-10000.0, 10000.0)
    y0 = np.random.uniform(-10000.0, 10000.0)
    z0 = np.random.uniform(2000.0, 8000.0)  # altitude in meters
    vg0 = np.random.uniform(180.0, 260.0)  # ground speed in m/s (~650 - 950 km/h)
    theta0 = np.random.uniform(0.0, 360.0)  # heading in degrees

    x = np.zeros(num_steps)
    y = np.zeros(num_steps)
    z = np.zeros(num_steps)
    vg = np.zeros(num_steps)
    theta = np.zeros(num_steps)

    x[0], y[0], z[0], vg[0], theta[0] = x0, y0, z0, vg0, theta0

    # Maneuver-specific control inputs
    if maneuver_class == 0:
        # Straight Cruise: Minimal heading/speed/altitude changes
        omega_base = np.random.uniform(-0.1, 0.1)
        at_base = np.random.uniform(-0.1, 0.1)
        vz_base = np.random.uniform(-0.5, 0.5)

        for t in range(1, num_steps):
            theta[t] = (theta[t - 1] + omega_base * dt + np.random.normal(0, 0.05)) % 360.0
            vg[t] = max(50.0, vg[t - 1] + at_base * dt + np.random.normal(0, 0.2))
            z[t] = max(100.0, z[t - 1] + vz_base * dt + np.random.normal(0, 0.5))

    elif maneuver_class == 1:
        # Coordinated Turn: Standard rate turn |omega| in [1.5, 4.0] deg/s, level flight
        turn_dir = np.random.choice([-1.0, 1.0])
        omega_turn = turn_dir * np.random.uniform(1.8, 3.8)
        at_base = np.random.uniform(-0.2, 0.2)
        vz_base = np.random.uniform(-0.5, 0.5)

        for t in range(1, num_steps):
            theta[t] = (theta[t - 1] + omega_turn * dt + np.random.normal(0, 0.1)) % 360.0
            vg[t] = max(50.0, vg[t - 1] + at_base * dt + np.random.normal(0, 0.2))
            z[t] = max(100.0, z[t - 1] + vz_base * dt + np.random.normal(0, 0.5))

    elif maneuver_class == 2:
        # Climb: vz in [+8, +25] m/s, az > 0 initially
        vz_target = np.random.uniform(10.0, 25.0)
        omega_base = np.random.uniform(-0.2, 0.2)

        for t in range(1, num_steps):
            theta[t] = (theta[t - 1] + omega_base * dt + np.random.normal(0, 0.05)) % 360.0
            # Climbing reduces kinetic speed slightly unless power added
            vg[t] = max(50.0, vg[t - 1] + np.random.uniform(-0.5, 0.1) * dt)
            current_vz = min(vz_target, 2.0 * t)  # Smooth transition to target climb rate
            z[t] = z[t - 1] + current_vz * dt + np.random.normal(0, 0.5)

    elif maneuver_class == 3:
        # Descent / Dive: vz in [-30, -10] m/s, az < 0
        vz_target = np.random.uniform(-25.0, -10.0)
        omega_base = np.random.uniform(-0.2, 0.2)

        for t in range(1, num_steps):
            theta[t] = (theta[t - 1] + omega_base * dt + np.random.normal(0, 0.05)) % 360.0
            # Diving increases ground speed slightly
            vg[t] = vg[t - 1] + np.random.uniform(0.1, 0.8) * dt
            current_vz = max(vz_target, -2.0 * t)
            z[t] = max(50.0, z[t - 1] + current_vz * dt + np.random.normal(0, 0.5))

    elif maneuver_class == 4:
        # Orbit / Holding Pattern: Continuous 360 deg turn loop
        orbit_dir = np.random.choice([-1.0, 1.0])
        omega_orbit = orbit_dir * np.random.uniform(2.5, 3.5)  # Completes circle in ~100-140 sec
        vz_base = np.random.uniform(-0.2, 0.2)

        for t in range(1, num_steps):
            theta[t] = (theta[t - 1] + omega_orbit * dt + np.random.normal(0, 0.08)) % 360.0
            vg[t] = max(50.0, vg[t - 1] + np.random.normal(0, 0.15))
            z[t] = max(100.0, z[t - 1] + vz_base * dt + np.random.normal(0, 0.3))

    elif maneuver_class == 5:
        # Evasive Maneuver: High G, sharp erratic turn (|omega| > 6.0 deg/s), high |at| > 3.0 m/s^2
        turn_dir = np.random.choice([-1.0, 1.0])
        omega_evasive = turn_dir * np.random.uniform(6.5, 12.0)
        at_evasive = np.random.uniform(3.5, 7.0) * np.random.choice([-1.0, 1.0])
        vz_evasive = np.random.uniform(-20.0, 20.0)

        for t in range(1, num_steps):
            # Dynamic flip of maneuver control inputs to simulate aggressive jinking
            if t % 8 == 0:
                omega_evasive = -omega_evasive * np.random.uniform(0.8, 1.2)
                at_evasive = -at_evasive

            theta[t] = (theta[t - 1] + omega_evasive * dt + np.random.normal(0, 0.2)) % 360.0
            vg[t] = max(80.0, min(350.0, vg[t - 1] + at_evasive * dt + np.random.normal(0, 0.5)))
            z[t] = max(100.0, z[t - 1] + vz_evasive * dt + np.random.normal(0, 1.0))

    # Integrate positions (x, y) based on ground speed and heading
    for t in range(1, num_steps):
        rad = math.radians(theta[t - 1])
        # Heading 0 deg = North (+Y), 90 deg = East (+X)
        dx = vg[t - 1] * math.sin(rad) * dt
        dy = vg[t - 1] * math.cos(rad) * dt
        x[t] = x[t - 1] + dx
        y[t] = y[t - 1] + dy

    # Inject sensor noise (Radar jitter)
    x_noisy = x + np.random.normal(0, noise_std * 10.0, size=num_steps)
    y_noisy = y + np.random.normal(0, noise_std * 10.0, size=num_steps)
    z_noisy = z + np.random.normal(0, noise_std * 5.0, size=num_steps)
    vg_noisy = vg + np.random.normal(0, noise_std * 2.0, size=num_steps)
    theta_noisy = (theta + np.random.normal(0, noise_std * 1.5, size=num_steps)) % 360.0

    # Compute derived kinematic features
    omega = np.zeros(num_steps)
    at = np.zeros(num_steps)
    az = np.zeros(num_steps)

    for t in range(1, num_steps):
        d_theta = normalize_angle_diff_deg(theta_noisy[t] - theta_noisy[t - 1])
        omega[t] = d_theta / dt

        d_vg = vg_noisy[t] - vg_noisy[t - 1]
        at[t] = d_vg / dt

        d_z = z_noisy[t] - z_noisy[t - 1]
        vz_curr = d_z / dt
        if t > 1:
            vz_prev = (z_noisy[t - 1] - z_noisy[t - 2]) / dt
            az[t] = (vz_curr - vz_prev) / dt
        else:
            az[t] = 0.0

    omega[0] = omega[1]
    at[0] = at[1]
    az[0] = az[1]

    # Combine into 8-feature matrix (T, 8)
    features = np.column_stack([x_noisy, y_noisy, z_noisy, vg_noisy, theta_noisy, omega, at, az])
    return features, maneuver_class


def build_dataset(
    samples_per_class: int = 1000,
    window_size: int = 30,
    output_dir: Path = Path("data/processed"),
    val_ratio: float = 0.2,
) -> None:
    """
    Generates balanced multi-class dataset, extracts 30-step windows,
    computes global Z-Score normalization parameters, and saves dataset artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[*] Generating {samples_per_class * 6} total trajectories (6 classes x {samples_per_class})...")

    all_windows = []
    all_labels = []

    for cls_id in range(6):
        cls_name = CLASS_NAMES[cls_id]
        print(f"  -> Generating Class {cls_id}: {cls_name}...")
        for _ in range(samples_per_class):
            traj, label = generate_single_trajectory(maneuver_class=cls_id, duration_sec=35, dt=1.0)
            # Extract sliding window of length 30
            # Taking the final 30 seconds of the trajectory
            window = traj[-window_size:, :]  # Shape: (30, 8)
            all_windows.append(window)
            all_labels.append(label)

    X = np.array(all_windows, dtype=np.float32)  # Shape: (N, 30, 8)
    y = np.array(all_labels, dtype=np.int64)    # Shape: (N,)

    # Shuffle dataset
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Split Train / Validation
    val_count = int(len(X) * val_ratio)
    train_X, val_X = X[val_count:], X[:val_count]
    train_y, val_y = y[val_count:], y[:val_count]

    # Calculate Normalization Parameters (Z-Score: mean & std per feature across all timesteps)
    # train_X has shape (N_train, 30, 8). Flatten over batch and time to compute (8,) mean and std
    flat_train = train_X.reshape(-1, 8)
    mean_vals = np.mean(flat_train, axis=0)
    std_vals = np.std(flat_train, axis=0)
    # Prevent division by zero
    std_vals = np.where(std_vals < 1e-6, 1.0, std_vals)

    norm_params = {
        "features": ["x", "y", "z", "vg", "theta", "omega", "at", "az"],
        "mean": mean_vals.tolist(),
        "std": std_vals.tolist(),
    }

    # Normalize datasets
    train_X_norm = (train_X - mean_vals) / std_vals
    val_X_norm = (val_X - mean_vals) / std_vals

    # Save outputs
    np.save(output_dir / "train_data.npy", train_X_norm)
    np.save(output_dir / "train_labels.npy", train_y)
    np.save(output_dir / "val_data.npy", val_X_norm)
    np.save(output_dir / "val_labels.npy", val_y)

    with open(output_dir / "norm_params.json", "w", encoding="utf-8") as f:
        json.dump(norm_params, f, indent=2)

    print("\n[+] Dataset generation complete!")
    print(f"  -> Train Samples: {train_X_norm.shape[0]} windows, shape: {train_X_norm.shape}")
    print(f"  -> Val Samples:   {val_X_norm.shape[0]} windows, shape: {val_X_norm.shape}")
    print(f"  -> Saved files in: {output_dir.resolve()}")
    print(f"  -> Z-Score Mean: {mean_vals.round(3)}")
    print(f"  -> Z-Score Std:  {std_vals.round(3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aero-Sense Synthetic Kinematic Dataset Generator")
    parser.add_argument("--samples_per_class", type=int, default=1200, help="Number of trajectories per maneuver class")
    parser.add_argument("--window_size", type=int, default=30, help="Sliding window size in seconds")
    parser.add_argument("--output_dir", type=str, default="data/processed", help="Directory to save .npy and .json artifacts")
    args = parser.parse_args()

    build_dataset(
        samples_per_class=args.samples_per_class,
        window_size=args.window_size,
        output_dir=Path(args.output_dir),
    )
