import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..guidance.pursuer import Pursuer
from ..guidance.evader import Evader
from ..kinematics.dynamics import derivative
from ..kinematics.integrator import rk4_step


COLUMNS: list[str] = [
    "time",
    "pursuer_x", "pursuer_y", "pursuer_z",
    "pursuer_vx", "pursuer_vy", "pursuer_vz",
    "evader_x", "evader_y", "evader_z",
    "evader_vx", "evader_vy", "evader_vz",
]


def closest_approach(
        previous_offset: NDArray[np.float64],
        current_offset: NDArray[np.float64]
) -> float:
    """
        Returns the smallest distance between both aircrafts during one time
        step, treating their relative motion as a straight line across the step.
        Checking only the sampled points lets fast aircrafts pass through each
        other between two steps without the interception being detected.
    """
    step: NDArray[np.float64] = current_offset - previous_offset
    length: float = float(np.dot(step, step))

    if length == 0.0:
        return float(np.linalg.norm(current_offset))

    fraction: float = float(np.clip(-np.dot(previous_offset, step) / length, 0.0, 1.0))

    return float(np.linalg.norm(previous_offset + fraction * step))


def run_simulation(
        pursuer: Pursuer,
        evader: Evader,
        dt: float,
        num_steps: int,
        intercept_radius: float
) -> pd.DataFrame:
    """
        Steps the joint pursuer and evader state forward with RK4 and records
        the time, position and velocity of both aircrafts at every step. The run
        ends early once the distance between both aircrafts falls below the
        intercept radius, which is labeled a hit in the outcome column and marks
        that final row with intercepted = 1, otherwise it runs for the full
        number of steps and is labeled a miss.
    """
    trajectory: list[NDArray[np.float64]] = []
    hit: bool = False

    joint_states: NDArray[np.float64] = np.concatenate(
        [pursuer.state, evader.state]
    )

    trajectory.append(np.concatenate([[0.0], joint_states]).copy())

    for step in range(num_steps):
        joint_states = rk4_step(
            dt,
            joint_states,
            pursuer,
            evader,
            derivative
        )

        pursuer.state = joint_states[:6]
        evader.state = joint_states[6:]
        time = (step + 1) * dt

        data_row: NDArray[np.float64] = np.concatenate(
            [[time], pursuer.state, evader.state]
        )

        trajectory.append(data_row.copy())

        previous_offset: NDArray[np.float64] = trajectory[-2][7:10] - trajectory[-2][1:4]
        current_offset: NDArray[np.float64] = evader.state[:3] - pursuer.state[:3]

        if closest_approach(previous_offset, current_offset) < intercept_radius:
            hit = True
            break

    data: pd.DataFrame = pd.DataFrame(trajectory, columns=COLUMNS)
    data["outcome"] = "hit" if hit else "miss"
    data["intercepted"] = 0

    if hit:
        data.loc[data.index[-1], "intercepted"] = 1

    data["destination_x"] = evader.destination[0]
    data["destination_y"] = evader.destination[1]
    data["destination_z"] = evader.destination[2]

    return data
