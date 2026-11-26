import numpy as np
from numpy.typing import NDArray

from ..kinematics.aircraft import Aircraft


class Pursuer(Aircraft):
    def __init__(
            self,
            position: NDArray[np.float64],
            velocity: NDArray[np.float64],
            navigation_constant: float,
            max_acceleration: float,
            seeker_half_angle: float
    ) -> None:
        super().__init__(position, velocity)
        self.navigation_constant = navigation_constant
        self.max_acceleration = max_acceleration
        self.seeker_half_angle = np.radians(seeker_half_angle)


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

        line_of_sight: NDArray[np.float64] = relative_position / relative_position_magnitude
        heading_error: float = np.arccos(np.clip(np.dot(line_of_sight, pursuer_unit_vector), -1.0, 1.0))

        if closing_speed <= 0 or heading_error > self.seeker_half_angle:
            """
                Proportional navigation only holds while the distance is closing and
                the evader is roughly ahead, otherwise a negative closing speed flips
                the command and a near zero one barely turns the pursuer. Until the
                evader is inside the seeker half angle the pursuer turns as hard as
                it can toward the line of sight.
            """
            turn_direction: NDArray[np.float64] = self.perpendicular_unit(
                line_of_sight,
                pursuer_unit_vector
            )

            return self.max_acceleration * turn_direction

        a_PN: NDArray[np.float64] = (
            self.navigation_constant
            *
            closing_speed
        ) * (
            np.cross(omega_LOS, pursuer_unit_vector)
        )

        return self.limit_acceleration(a_PN, self.max_acceleration)


