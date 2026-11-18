import numpy as np
from numpy.typing import NDArray
import pandas as pd

from ..guidance.pursuer import Pursuer
from ..guidance.evader import Evader
from ..kinematics.dynamics import derivative
from ..kinematics.integrator import rk4_step


def run_simulation(
        pursuer: Pursuer,
        evader: Evader,
        dt: float,
        num_steps: int
) -> pd.DataFrame:
    trajectory: list[NDArray[np.float64]] = []
    
    joint_states: NDArray[np.float64] = np.concatenate(
        [pursuer.state, evader.state]
    )

    trajectory.append(np.concatenate([0.0, joint_states]).copy())

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
            [time, pursuer.state, evader.state]
        )

        trajectory.append(data_row.copy())

    return trajectory



