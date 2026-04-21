# dronebot-sim

Drone base simulation and control with MuJoCo.

---

## Week 1 Plan: Falling Body → Hover

### Day 1: Repo + Model Skeleton
- Create repo and folder structure
- Add one MuJoCo XML with free-body drone base
- Run one step and log `qpos`/`qvel`

**Exit criteria:** gravity causes z drop; sim steps reliably.

---

### Day 2: Motor Force/Torque Mixer
Implement 4-motor mapping:
- Total thrust
- Roll/pitch torques from arm leverage
- Yaw torque from rotor drag sign
- Add actuator clamps

**Exit criteria:** commanded thrust changes altitude; torques rotate body in expected direction.

---

### Day 3: Altitude Hold (inner-most success)
- PID on z (or vertical velocity + z outer)
- Tune with no wind/disturbances first

**Exit criteria:** hover within ±10 cm for 10–20 s.

---

### Day 4: Attitude Hold
- Add roll/pitch/yaw control loops
- Tune at small-angle commands

**Exit criteria:** angle command tracking without oscillation or saturation.

---

### Day 5: Position Hold
- Outer loop (x, y, z) → attitude setpoints + thrust
- Keep limits on tilt angle

**Exit criteria:** hold a point; recover from small disturbance.

---

### Day 6: Disturbance + Realism Pass
- Add sensor noise/bias + simple latency
- Add mass/inertia from BOM approximations
- Add scripted gust impulses

**Exit criteria:** still stable with mild disturbances and noisy state.

---

### Day 7: Validation Harness
Add repeatable eval script + plots:
- RMS position error
- Settling time
- Control effort / saturation %
- Save run summaries to JSON/CSV

**Exit criteria:** one command gives reproducible metrics.

---

## Controller Order (important)

Build and tune in this order — do not start with full 6-DOF:

1. Thrust / altitude
2. Attitude
3. Position

Tuning all loops simultaneously is painful and converges slowly.

---

## BOM → Sim Params Quick Mapping

From your BOM CSV, estimate the following sim parameters:

| Parameter | Source |
|---|---|
| `mass_kg` | Total assembled mass |
| `arm_length_m` | Center to motor distance |
| `Ixx, Iyy, Izz` | Rough CAD or box/cylinder approximation |
| Thrust margin | Hover thrust should be ~40–60% of max thrust |

---

## Installation

```bash
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.10
- MuJoCo >= 3.1.0
