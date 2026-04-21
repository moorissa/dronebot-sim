#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml


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


def rotor_table_from_cfg(cfg: dict) -> list[dict]:
    """Return rotor definitions from config or fallback to demo quad."""
    rotors = cfg.get("rotors")
    if isinstance(rotors, list) and rotors:
        out: list[dict] = []
        for r in rotors:
            out.append(
                {
                    "name": str(r["name"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "spin": float(r.get("spin", 1.0)),
                }
            )
        return out
    # Backward-compatible demo quad.
    arm = float(cfg.get("arm_length_m", 0.18))
    return [
        {"name": "rotor_fl", "x": arm, "y": arm, "spin": 1.0},
        {"name": "rotor_fr", "x": arm, "y": -arm, "spin": -1.0},
        {"name": "rotor_rl", "x": -arm, "y": arm, "spin": -1.0},
        {"name": "rotor_rr", "x": -arm, "y": -arm, "spin": 1.0},
    ]


def quat_to_rpy_wxyz(q: np.ndarray) -> tuple[float, float, float]:
    # q = [w, x, y, z]
    w, x, y, z = q
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = float(np.arctan2(sinr_cosp, cosr_cosp))

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = float(np.sign(sinp) * (np.pi / 2))
    else:
        pitch = float(np.arcsin(sinp))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = float(np.arctan2(siny_cosp, cosy_cosp))
    return roll, pitch, yaw


def motor_mix_general(
    total_thrust: float,
    tau_roll: float,
    tau_pitch: float,
    tau_yaw: float,
    rotors: list[dict],
    yaw_moment_coeff: float,
) -> np.ndarray:
    """Least-squares thrust allocation for arbitrary rotor layout.

    Each rotor i contributes:
      Fz: +fi
      tau_x: y_i * fi
      tau_y: -x_i * fi
      tau_z: (k_yaw * spin_i) * fi
    """
    n = len(rotors)
    a = np.zeros((4, n), dtype=float)
    for i, r in enumerate(rotors):
        x = float(r["x"])
        y = float(r["y"])
        s = float(r.get("spin", 1.0))
        a[:, i] = np.array([1.0, y, -x, yaw_moment_coeff * s], dtype=float)
    b = np.array([total_thrust, tau_roll, tau_pitch, tau_yaw], dtype=float)
    return np.linalg.pinv(a) @ b


def run_demo(model_path: Path, cfg_path: Path, viewer: bool) -> int:
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)

    mass = float(cfg["mass_kg"])
    max_motor = float(cfg["max_thrust_per_motor_n"])
    z_ref = float(cfg["hover_target_z_m"])
    yaw_moment_coeff = float(cfg.get("yaw_moment_coeff", 0.02))
    rotors = rotor_table_from_cfg(cfg)
    duration_s = float(cfg.get("duration_s", 10.0))
    log_hz = float(cfg.get("log_hz", 20.0))
    gains = Gains(
        z_kp=float(cfg["z_kp"]),
        z_ki=float(cfg["z_ki"]),
        z_kd=float(cfg["z_kd"]),
        att_kp_roll=float(cfg["att_kp_roll"]),
        att_kp_pitch=float(cfg["att_kp_pitch"]),
        att_kp_yaw=float(cfg["att_kp_yaw"]),
        att_kd_p=float(cfg["att_kd_p"]),
        att_kd_q=float(cfg["att_kd_q"]),
        att_kd_r=float(cfg["att_kd_r"]),
    )

    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "drone")
    if bid < 0:
        raise RuntimeError("Body 'drone' not found in model.")

    dt = m.opt.timestep
    steps = int(duration_s / dt)
    log_every = max(1, int((1.0 / log_hz) / dt))

    # Integrator state
    z_int = 0.0
    z_prev = d.qpos[2]

    g = 9.81
    print("Starting hover demo...")
    print(f"dt={dt:.4f}s steps={steps} mass={mass}kg max_motor={max_motor}N rotors={len(rotors)}")
    print(f"qpos before step: {d.qpos[:7]}")

    vctx = None
    if viewer:
        from mujoco import viewer as mj_viewer

        vctx = mj_viewer.launch_passive(m, d)

    for k in range(steps):
        # State
        z = float(d.qpos[2])
        q = np.array(d.qpos[3:7], dtype=float)
        p, q_rate, r = [float(x) for x in d.qvel[3:6]]
        roll, pitch, yaw = quat_to_rpy_wxyz(q / max(np.linalg.norm(q), 1e-9))

        # Altitude PID
        z_err = z_ref - z
        z_dot = (z - z_prev) / dt
        z_prev = z
        z_int += z_err * dt
        z_int = float(np.clip(z_int, -1.0, 1.0))
        u_z = gains.z_kp * z_err + gains.z_ki * z_int - gains.z_kd * z_dot
        total_thrust = mass * g + u_z
        total_thrust = float(np.clip(total_thrust, 0.0, 4.0 * max_motor))

        # Small-angle attitude stabilizer to keep level + yaw ~ 0
        tau_roll = -gains.att_kp_roll * roll - gains.att_kd_p * p
        tau_pitch = -gains.att_kp_pitch * pitch - gains.att_kd_q * q_rate
        tau_yaw = -gains.att_kp_yaw * yaw - gains.att_kd_r * r

        motors = motor_mix_general(
            total_thrust,
            tau_roll,
            tau_pitch,
            tau_yaw,
            rotors=rotors,
            yaw_moment_coeff=yaw_moment_coeff,
        )
        motors = np.clip(motors, 0.0, max_motor)

        # Apply rotor forces in body frame at each rotor site.
        # Positive force in local +Z.
        for rotor, thrust in zip(rotors, motors):
            sid_name = rotor["name"]
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sid_name)
            if sid < 0:
                continue
            force_torque = np.array([0.0, 0.0, float(thrust), 0.0, 0.0, 0.0], dtype=float)
            mujoco.mj_applyFT(m, d, force_torque[:3], force_torque[3:], d.site_xpos[sid], bid, d.qfrc_applied)

        mujoco.mj_step(m, d)
        d.qfrc_applied[:] = 0.0

        if vctx is not None and vctx.is_running():
            vctx.sync()
        if k % log_every == 0 or k == steps - 1:
            print(
                f"t={k * dt:5.2f}s z={z:6.3f} roll={roll:+.3f} pitch={pitch:+.3f} "
                f"yaw={yaw:+.3f} thrust={total_thrust:6.2f}N motors={np.round(motors, 2)}"
            )

    if vctx is not None:
        vctx.close()

    print(f"qpos after step:  {d.qpos[:7]}")
    print("Done.")
    return 0


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="MuJoCo drone-base hover demo.")
    ap.add_argument("--model", type=Path, default=repo / "models" / "drone_base.xml")
    ap.add_argument("--config", type=Path, default=repo / "configs" / "drone_base.yaml")
    ap.add_argument("--viewer", action="store_true", help="Show MuJoCo passive viewer.")
    args = ap.parse_args()

    model = args.model if args.model.is_absolute() else (repo / args.model).resolve()
    config = args.config if args.config.is_absolute() else (repo / args.config).resolve()
    if not model.is_file():
        raise SystemExit(f"Missing model: {model}")
    if not config.is_file():
        raise SystemExit(f"Missing config: {config}")

    return run_demo(model, config, args.viewer)


if __name__ == "__main__":
    raise SystemExit(main())
