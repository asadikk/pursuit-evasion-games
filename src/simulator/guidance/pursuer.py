import numpy as np
from numpy.typing import NDArray

from ..kinematics.aircraft import Aircraft


class Pursuer(Aircraft):
    def __init__(
            self,
            position: NDArray[np.float64],
            velocity: NDArray[np.float64],
            navigation_constant: float
    ) -> None:
        super().__init__(position, velocity)
        self.navigation_constant = navigation_constant


    def relative_position(
            self,
            pursuer_position: NDArray[np.float64],
            evader_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return evader_position - pursuer_position


    def relative_velocity(
            self,
            pursuer_velocity: NDArray[np.float64],
            evader_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return evader_velocity - pursuer_velocity


    def relative_position_magnitude(
            self,
            relative_position: NDArray[np.float64]
    ) -> float:
        """
            Returns the magnitude of the relative position vector which
            represents the distance between the pursuer and evader aircrafts.
        """

        return np.linalg.norm(relative_position)


    def closing_speed(
            self,
            relative_position: NDArray[np.float64],
            relative_velocity: NDArray[np.float64],
            relative_position_magnitude: float
    ) -> float:
        """
            The resulting float value returned from this function is
            positive when the 'relative_position_magnitude' or the distance
            between both aircrafts is decreasing and negative when the distance
            between both aircrafts is increasing.
        """
        temp: float = -np.dot(relative_position, relative_velocity)
        return temp / relative_position_magnitude


    def omega_LOS(
            self,
            relative_position: NDArray[np.float64],
            relative_velocity: NDArray[np.float64],
            relative_position_magnitude: float
    ) -> NDArray[np.float64]:
        """
            This returns the angular velocity of the line of sight vector which
            points from the pursuer to the evader, it tells us how fast the vector
            is rotating as re-directs its pointer toward the evader.
        """
        temp: NDArray[np.float64] = np.cross(relative_position, relative_velocity)
        return temp / np.square(relative_position_magnitude)


    def pursuer_unit_vector(
            self,
            pursuer_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return pursuer_velocity / np.linalg.norm(pursuer_velocity)


    def guidance(
            self,
            pursuer_state: NDArray[np.float64],
            evader_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        relative_position: NDArray[np.float64] = self.relative_position(pursuer_state[:3], evader_state[:3])
        relative_velocity: NDArray[np.float64] = self.relative_velocity(pursuer_state[3:], evader_state[3:])
        relative_position_magnitude: float = self.relative_position_magnitude(relative_position)

        closing_speed: float = self.closing_speed(relative_position, relative_velocity, relative_position_magnitude)
        omega_LOS: NDArray[np.float64] = self.omega_LOS(relative_position, relative_velocity, relative_position_magnitude)
        pursuer_unit_vector: NDArray[np.float64] = self.pursuer_unit_vector(pursuer_state[3:])

        a_PN: NDArray[np.float64] = (
            self.navigation_constant
            *
            closing_speed
        ) * (
            np.cross(omega_LOS, pursuer_unit_vector)
        )

        return a_PN


