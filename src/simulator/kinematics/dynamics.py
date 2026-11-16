import numpy as np
from numpy.typing import NDArray

from ..guidance.pursuer import Pursuer
from ..guidance.evader import Evader


def derivative(
    joint_states: NDArray[np.float64],
    pursuer: Pursuer,
    evader: Evader,
    ) -> NDArray[np.float64]:
    pursuer_state: NDArray[np.float64] = joint_states[:6]
    evader_state: NDArray[np.float64] = joint_states[6:]

    pursuer_acceleration: NDArray[np.float64] = pursuer.guidance(pursuer_state, evader_state)
    evader_acceleration: NDArray[np.float64] = evader.guidance(pursuer_state, evader_state)

    pursuer_velocity: NDArray[np.float64] = pursuer_state[3:]
    evader_velocity: NDArray[np.float64] = evader_state[3:]

    return np.concatenate([
        pursuer_velocity,
        pursuer_acceleration,
        evader_velocity,
        evader_acceleration
    ])
    