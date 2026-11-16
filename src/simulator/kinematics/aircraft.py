"""
    Herein lies the aircraft vehicle class which will define and calculate
    how a 3D point-mass aircraft vehicle model operates within the domain of representing a 
    simplified aircraft with arbitrary aerodynamic coefficients, and merely reflects kinematic motion.

    x - Position along the world X-axis (m)
    y - Position along the world Y-axis (m)
    z - Altitude, vertical position along the world Z-axis (m)
    v_x - Velocity component along the world X-axis (m/s)
    v_y - Velocity component along the world Y-axis (m/s)
    v_z - Velocity component along the world Z-axis (m/s)

    state = [x, y, z, v_x, v_y, v_z]
"""


import numpy as np
from numpy.typing import NDArray


class Aircraft:
    def __init__(self, position: NDArray[np.float64], velocity: NDArray[np.float64]) -> None:
        if position.shape != (3,) and velocity.shape != (3,):
            raise ValueError("Position and velocity vectors must be 1D and contain 3 float values.")

        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("All position and velocity values must be finite float values.")

        speed: float = np.linalg.norm(velocity)

        self.initial_speed = float(speed)

        self.state: NDArray[np.float64] = np.concatenate([position, velocity])


