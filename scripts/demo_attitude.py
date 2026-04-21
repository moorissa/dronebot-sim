#!/usr/bin/env python3
"""Step 4: Attitude hold demo.

Steps through a sequence of roll/pitch/yaw commands while maintaining altitude.
Exit criteria: angle command tracking without oscillation or saturation.

Run:
    python3 scripts/demo_attitude.py --model models/drone_hex_base.xml \
        --config configs/drone_hex_ndaa_budget.yaml
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Shared helpers (duplicated from demo_hover for standalone use)
# ---------------------------------------------------------------------------

@dataclass
class Gains:
    z_kp: float
    z_ki: float
    z_kd: float
    att_kp_roll: float
    att_kp_pitch: float
    att_kp_yaw: float
    att_kd_p: float
    att_kd_q: float
    att_kd_r: float


@dataclass
class AttCmd:
    """A timed attitude command."""
    t_start: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


# Default command sequence (seconds, deg).
DEFAULT_SEQUENCE: list[AttCmd] = [
    AttCmd(0.0,   0.0,  0.0,  0.0),
    AttCmd(2.0,  10.0,  0.0,  0.0),   # +10° roll
    AttCmd(4.0,   0.0,  0.0,  0.0),   # back to level
    AttCmd(6.0,   0.0, 10.0,  0.0),   # +10° pitch
    AttCmd(8.0,   0.0,  0.0,  0.0),   # back to level
    AttCmd(10.0,  0.0,  0.0, 20.0),   # +20° yaw
    AttCmd(13.0,  0.0,  0.0,  0.0),   # back to level
]


def rotor_table_from_cfg(cfg: dict) -> list[dict]:
    rotors = cfg.get("rotors")
    if isinstance(rotors, list) and rotors:
        return [
            {"name": str(r["name"]), "x": float(r["x"]),
             "y": float(r["y"]), "spin": float(r.get("spin", 1.0))}
            for r in rotors
        ]
    arm = float(cfg.get("arm_length_m", 0.18))
    return [
        {"name": "rotor_fl", "x":  arm, "y":  arm, "spin":  1.0},
        {"name": "rotor_fr", "x":  arm, "y": -arm, "spin": -1.0},
        {"name": "rotor_rl", "x": -arm, "y":  arm, "spin": -1.0},
        {"name": "rotor_rr", "x": -arm, "y": -arm, "spin":  1.0},
    ]


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = float(np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y)))
    sinp = 2*(w*y - z*x)
    pitch = float(np.sign(sinp) * np.pi/2 if abs(sinp) >= 1 else np.arcsin(sinp))
    yaw = float(np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)))
    return roll, pitch, yaw


def motor_mix(
    total_thrust: float, tau_roll: float, tau_pitch: float, tau_yaw: float,
    rotors: list[dict], k_yaw: float,
) -> np.ndarray:
    n = len(rotors)
    a = np.zeros((4, n))
    for i, r in enumerate(rotors):
        a[:, i] = [1.0, r["y"], -r["x"], k_yaw * r["spin"]]
    b = np.array([total_thrust, tau_roll, tau_pitch, tau_yaw])
    return np.linalg.pinv(a) @ b


def current_cmd(t: float, sequence: list[AttCmd]) -> AttCmd:
    """Return the active command at time t."""
    active = sequence[0]
    for cmd in sequence:
        if t >= cmd.t_start:
            active = cmd
    return active


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def run(model_path: Path, cfg_path: Path, viewer: bool, max_tilt_deg: float) -> int:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    mass        = float(cfg["mass_kg"])
    max_motor   = float(cfg["max_thrust_per_motor_n"])
    z_ref       = float(cfg["hover_target_z_m"])
    k_yaw       = float(cfg.get("yaw_moment_coeff", 0.02))
    rotors      = rotor_table_from_cfg(cfg)
    duration_s  = float(cfg.get("att_demo_duration_s", cfg.get("duration_s", 16.0)))
    log_hz      = float(cfg.get("log_hz", 20.0))

    gains = Gains(
        z_kp=float(cfg["z_kp"]), z_ki=float(cfg["z_ki"]), z_kd=float(cfg["z_kd"]),
        att_kp_roll=float(cfg["att_kp_roll"]), att_kp_pitch=float(cfg["att_kp_pitch"]),
        att_kp_yaw=float(cfg["att_kp_yaw"]),
        att_kd_p=float(cfg["att_kd_p"]), att_kd_q=float(cfg["att_kd_q"]),
        att_kd_r=float(cfg["att_kd_r"]),
    )

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    if bid < 0:
        raise RuntimeError("Body 'drone' not found in model.")

    dt        = m.opt.timestep
    steps     = int(duration_s / dt)
    log_every = max(1, int((1.0 / log_hz) / dt))
    max_tilt  = np.radians(max_tilt_deg)

    # Integrator state
    z_int  = 0.0
    z_prev = float(d.qpos[2])
    g      = 9.81

    # Metrics
    att_errors: list[float] = []
    saturated_steps = 0

    print("=" * 60)
    print("Step 4 — Attitude Hold Demo")
    print(f"  model : {model_path.name}")
    print(f"  config: {cfg_path.name}")
    print(f"  mass={mass}kg  rotors={len(rotors)}  max_tilt=±{max_tilt_deg}°")
    print(f"  duration={duration_s}s  dt={dt*1000:.1f}ms")
    print("=" * 60)
    print(f"{'t':>6} {'roll_cmd':>9} {'pitch_cmd':>10} {'yaw_cmd':>8} "
          f"{'roll':>7} {'pitch':>7} {'yaw':>7} {'err_deg':>8} {'thrust':>8}")

    vctx = None
    if viewer:
        from mujoco import viewer as mj_viewer
        vctx = mj_viewer.launch_passive(m, d)

    for k in range(steps):
        t = k * dt

        # Current attitude command
        cmd  = current_cmd(t, DEFAULT_SEQUENCE)
        r_ref = np.radians(cmd.roll_deg)
        p_ref = np.radians(cmd.pitch_deg)
        y_ref = np.radians(cmd.yaw_deg)

        # Clamp commanded tilt for safety
        r_ref = float(np.clip(r_ref, -max_tilt, max_tilt))
        p_ref = float(np.clip(p_ref, -max_tilt, max_tilt))

        # State
        z   = float(d.qpos[2])
        q   = d.qpos[3:7].copy()
        q  /= max(np.linalg.norm(q), 1e-9)
        p_rate, q_rate, r_rate = [float(x) for x in d.qvel[3:6]]
        roll, pitch, yaw = quat_to_rpy(q)

        # Altitude PID
        z_err  = z_ref - z
        z_dot  = (z - z_prev) / dt
        z_prev = z
        z_int  = float(np.clip(z_int + z_err * dt, -1.0, 1.0))
        u_z    = gains.z_kp * z_err + gains.z_ki * z_int - gains.z_kd * z_dot
        T      = float(np.clip(mass * g + u_z, 0.0, len(rotors) * max_motor))

        # Attitude PID — track commanded angles
        tau_roll  = gains.att_kp_roll  * (r_ref - roll)  - gains.att_kd_p * p_rate
        tau_pitch = gains.att_kp_pitch * (p_ref - pitch) - gains.att_kd_q * q_rate
        tau_yaw   = gains.att_kp_yaw   * (y_ref - yaw)  - gains.att_kd_r * r_rate

        motors = motor_mix(T, tau_roll, tau_pitch, tau_yaw, rotors, k_yaw)
        sat    = np.any(motors > max_motor) or np.any(motors < 0)
        motors = np.clip(motors, 0.0, max_motor)
        if sat:
            saturated_steps += 1

        # Apply rotor forces
        for rotor, thrust in zip(rotors, motors):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, rotor["name"])
            if sid < 0:
                continue
            mujoco.mj_applyFT(
                m, d,
                np.array([0.0, 0.0, float(thrust)]),
                np.zeros(3),
                d.site_xpos[sid], bid, d.qfrc_applied,
            )

        mujoco.mj_step(m, d)
        d.qfrc_applied[:] = 0.0

        if vctx is not None and vctx.is_running():
            vctx.sync()

        # Attitude tracking error (RMS over roll+pitch+yaw)
        err_rad = np.sqrt(
            (r_ref - roll)**2 + (p_ref - pitch)**2 + (y_ref - yaw)**2
        ) / np.sqrt(3)
        att_errors.append(np.degrees(err_rad))

        if k % log_every == 0 or k == steps - 1:
            print(
                f"{t:6.2f}s "
                f"{cmd.roll_deg:+9.1f}° "
                f"{cmd.pitch_deg:+10.1f}° "
                f"{cmd.yaw_deg:+8.1f}° "
                f"{np.degrees(roll):+7.2f}° "
                f"{np.degrees(pitch):+7.2f}° "
                f"{np.degrees(yaw):+7.2f}° "
                f"{np.degrees(err_rad):8.3f}° "
                f"{T:7.2f}N"
                + (" SAT" if sat else "")
            )

    if vctx is not None:
        vctx.close()

    # Summary
    arr = np.array(att_errors)
    sat_pct = 100.0 * saturated_steps / steps
    print()
    print("=" * 60)
    print("Summary")
    print(f"  RMS attitude error : {arr.mean():.3f}° (mean of per-step RMS)")
    print(f"  Max attitude error : {arr.max():.3f}°")
    print(f"  Saturation         : {sat_pct:.1f}% of steps")
    print("=" * 60)

    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Step 4: MuJoCo drone attitude hold demo.")
    ap.add_argument("--model",  type=Path, default=repo / "models" / "drone_hex_base.xml")
    ap.add_argument("--config", type=Path, default=repo / "configs" / "drone_hex_ndaa_budget.yaml")
    ap.add_argument("--viewer", action="store_true", help="Show MuJoCo passive viewer.")
    ap.add_argument("--max-tilt-deg", type=float, default=30.0,
                    help="Max commanded tilt angle in degrees (default: 30).")
    args = ap.parse_args()

    model  = args.model  if args.model.is_absolute()  else (repo / args.model).resolve()
    config = args.config if args.config.is_absolute() else (repo / args.config).resolve()
    if not model.is_file():
        raise SystemExit(f"Missing model: {model}")
    if not config.is_file():
        raise SystemExit(f"Missing config: {config}")

    return run(model, config, args.viewer, args.max_tilt_deg)


if __name__ == "__main__":
    raise SystemExit(main())
