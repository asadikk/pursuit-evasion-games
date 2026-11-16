import numpy as np
from numpy.typing import NDArray
from collections.abc import Callable

from ..guidance.pursuer import Pursuer
from ..guidance.evader import Evader


def rk4_step(
    dt: float,
    joint_states: NDArray[np.float64],
    pursuer: Pursuer,
    evader: Evader,
    derivative: Callable[
        [NDArray[np.float64]],
        NDArray[np.float64]
    ]
) -> NDArray[np.float64]:
    k1 = derivative(joint_states, pursuer, evader)
    k2 = derivative(joint_states + (dt/2) * k1, pursuer, evader)
    k3 = derivative(joint_states + (dt/2) * k2, pursuer, evader)
    k4 = derivative(joint_states + dt * k3, pursuer, evader)

    return joint_states + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
