import numpy as np
from numpy.typing import NDArray

from ..kinematics.aircraft import Aircraft


class Evader(Aircraft):
    def __init__(
            self,
            position: NDArray[np.float64],
            velocity: NDArray[np.float64],
            destination: NDArray[np.float64],
            evasion_gain: float,
            destination_gain: float
    ) -> None:
        super().__init__(position, velocity)
        self.destination = destination
        self.evasion_gain = evasion_gain
        self.destination_gain = destination_gain


    def evader_unit_vector(
            self,
            evader_velocity: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return evader_velocity / np.linalg.norm(evader_velocity)


    def guidance(
            self,
            pursuer_state: NDArray[np.float64],
            evader_state: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pursuer_position: NDArray[np.float64] = pursuer_state[:3]
        evader_position: NDArray[np.float64] = evader_state[:3]
        evader_velocity: NDArray[np.float64] = evader_state[3:]
        
        pursue_to_evade: NDArray[np.float64] = self._pursuer_to_evader(
            pursuer_position,
            evader_position
        )

        evade_to_dest: NDArray[np.float64] = self._evader_to_destination(
            evader_position
        )

        evader_unit_vector: NDArray[np.float64] = (
            evader_velocity
            /
            np.linalg.norm(evader_velocity)
        )

        evasion_vector: NDArray[np.float64] = self._perpendicular_comp(
            pursue_to_evade,
            evader_unit_vector
        )

        destination_vector: NDArray[np.float64] = self._perpendicular_comp(
            evade_to_dest,
            evader_unit_vector
        )

        return (
            (
                self.evasion_gain
                *
                evasion_vector
            )
            +
            (
                self.destination_gain
                *
                destination_vector
            )
        )


    def _pursuer_to_evader(
            self,
            pursuer_position: NDArray[np.float64],
            evader_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
            Returns a unit vector from the pursuer to the evader.
        """
        relative_position: NDArray[np.float64] = (
            evader_position
            -
            pursuer_position
        )

        return relative_position / np.linalg.norm(relative_position)


    def _evader_to_destination(
            self,
            evader_position: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
            Return a unit vector from the evader to the defined destination.
        """
        relative_position: NDArray[np.float64] = (
            self.destination
            -
            evader_position
        )

        return relative_position / np.linalg.norm(relative_position)


    def _perpendicular_comp(
            self,
            desired_unit_vector: NDArray[np.float64],
            evader_unit_vector: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
            This function returns the component of the desired direction that
            is perpendicular to the evader's current velocity.
        """
        return (
            desired_unit_vector
            -
            (
                np.dot(desired_unit_vector, evader_unit_vector)
                *
                evader_unit_vector
            )
        )
