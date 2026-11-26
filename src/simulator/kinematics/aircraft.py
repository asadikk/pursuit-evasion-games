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
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("Position and velocity vectors must be 1D and contain 3 float values.")

        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("All position and velocity values must be finite float values.")

        speed: float = np.linalg.norm(velocity)

        self.initial_speed = float(speed)

        self.state: NDArray[np.float64] = np.concatenate([position, velocity])


    def limit_acceleration(
            self,
            acceleration: NDArray[np.float64],
            max_acceleration: float
    ) -> NDArray[np.float64]:
        """
            Scales the commanded acceleration down to the maximum the aircraft
            can sustain while keeping its direction unchanged.
        """
        magnitude: float = np.linalg.norm(acceleration)

        if magnitude <= max_acceleration:
            return acceleration

        return acceleration * (max_acceleration / magnitude)


    def perpendicular_unit(
            self,
            desired_vector: NDArray[np.float64],
            unit_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
            Returns the unit vector of the part of the desired direction that is
            perpendicular to the current velocity, which is the direction the
            aircraft must turn toward. When the desired direction lies along the
            velocity a horizontal perpendicular is returned instead so the turn
            is never undefined.
        """
        perpendicular: NDArray[np.float64] = desired_vector - np.dot(desired_vector, unit_velocity) * unit_velocity
        magnitude: float = np.linalg.norm(perpendicular)

        if magnitude > 1e-6:
            return perpendicular / magnitude

        fallback: NDArray[np.float64] = np.cross(unit_velocity, np.array([0.0, 0.0, 1.0]))

        if np.linalg.norm(fallback) < 1e-6:
            fallback = np.cross(unit_velocity, np.array([1.0, 0.0, 0.0]))

        return fallback / np.linalg.norm(fallback)
