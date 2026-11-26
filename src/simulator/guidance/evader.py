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
            destination_gain: float,
            evasion_range: float,
            max_acceleration: float
    ) -> None:
        super().__init__(position, velocity)
        self.destination = destination
        self.evasion_gain = evasion_gain
        self.destination_gain = destination_gain
        self.evasion_range = evasion_range
        self.max_acceleration = max_acceleration


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

        evasion_vector: NDArray[np.float64] = self.perpendicular_unit(
            pursue_to_evade,
            evader_unit_vector
        )

        evasion_weight: float = self._evasion_weight(
            pursuer_position,
            evader_position
        )

        if self._reaches_destination_first(pursuer_state, evader_state):
            evasion_weight = 0.0

        destination_vector: NDArray[np.float64] = self._perpendicular_comp(
            evade_to_dest,
            evader_unit_vector
        )

        acceleration: NDArray[np.float64] = (
            (
                self.evasion_gain
                *
                evasion_weight
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

        return self.limit_acceleration(acceleration, self.max_acceleration)


    def _evasion_weight(
            self,
            pursuer_position: NDArray[np.float64],
            evader_position: NDArray[np.float64]
    ) -> float:
        """
            Returns a weight between 0 and 1 that scales the evasive maneuver,
            it is 0 while the pursuer is at or beyond the evasion range and
            grows linearly to 1 as the pursuer closes in on the evader.
        """
        distance: float = np.linalg.norm(evader_position - pursuer_position)

        return float(np.clip(1.0 - distance / self.evasion_range, 0.0, 1.0))


    def _reaches_destination_first(
            self,
            pursuer_state: NDArray[np.float64],
            evader_state: NDArray[np.float64]
    ) -> bool:
        """
            Returns True when the evader will reach its destination before the
            pursuer can close the distance to it, in which case evading would
            only throw away an arrival that is already secured.

            The evader's time to the destination is its distance to the
            destination divided by its speed. The pursuer's time to catch the
            evader is the distance between them divided by the rate at which
            that distance is currently shrinking, and it never catches the
            evader while the distance is not shrinking at all.
        """
        evader_position: NDArray[np.float64] = evader_state[:3]
        evader_velocity: NDArray[np.float64] = evader_state[3:]

        relative_position: NDArray[np.float64] = evader_position - pursuer_state[:3]
        relative_velocity: NDArray[np.float64] = evader_velocity - pursuer_state[3:]
        separation: float = float(np.linalg.norm(relative_position))

        closing_speed: float = -float(np.dot(relative_position, relative_velocity)) / separation

        if closing_speed <= 0.0:
            return True

        time_to_destination: float = float(np.linalg.norm(self.destination - evader_position)) / float(np.linalg.norm(evader_velocity))
        time_to_catch: float = separation / closing_speed

        return time_to_destination <= time_to_catch


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
