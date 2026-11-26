from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..guidance import Pursuer, Evader
from .simulation import run_simulation


RUNS_DIR: Path = Path(__file__).resolve().parents[1] / "runs"

TOLERANCE: float = 1e-6


def sample_position_at_distance(
        rng: np.random.Generator,
        destination: NDArray[np.float64],
        distance: float,
        altitude_bounds: NDArray[np.float64]
) -> NDArray[np.float64]:
    """
        Returns a point exactly the given distance away from the destination.
        The altitude is drawn uniformly within the altitude bounds and the
        bearing around the destination is drawn uniformly, the horizontal
        distance then follows so the straight-line distance is exact.
    """
    altitude: float = rng.uniform(altitude_bounds[0], altitude_bounds[1])
    height: float = altitude - destination[2]
    horizontal: float = np.sqrt(distance ** 2 - height ** 2)
    bearing: float = rng.uniform(0.0, 2 * np.pi)

    return destination + np.array([
        horizontal * np.cos(bearing),
        horizontal * np.sin(bearing),
        height
    ])


def sample_direction(rng: np.random.Generator) -> NDArray[np.float64]:
    """
        Returns a unit vector pointing in a uniformly random direction.
    """
    direction: NDArray[np.float64] = rng.normal(size=3)

    return direction / np.linalg.norm(direction)


def valid_start(
        pursuer_position: NDArray[np.float64],
        pursuer_velocity: NDArray[np.float64],
        evader_position: NDArray[np.float64],
        evader_velocity: NDArray[np.float64],
        destination: NDArray[np.float64],
        min_separation: float,
        min_destination_distance: float,
        evader_speed_ratio: float
) -> bool:
    """
        Returns True when the start is fair. Both aircrafts must be exactly the
        same distance away from the destination, that distance must be at least
        the minimum destination distance, they must start at least the minimum
        separation apart, and the evader must fly at exactly the evader speed
        ratio times the pursuer's speed.
    """
    pursuer_distance: float = np.linalg.norm(destination - pursuer_position)
    evader_distance: float = np.linalg.norm(destination - evader_position)
    pursuer_speed: float = np.linalg.norm(pursuer_velocity)
    evader_speed: float = np.linalg.norm(evader_velocity)

    return bool(
        abs(pursuer_distance - evader_distance) <= TOLERANCE * pursuer_distance
        and pursuer_distance >= min_destination_distance
        and np.linalg.norm(evader_position - pursuer_position) >= min_separation
        and abs(evader_speed - evader_speed_ratio * pursuer_speed) <= TOLERANCE * pursuer_speed
    )


def next_run_id() -> int:
    """
        Returns the next unused integer run id, one greater than the highest id
        already present in the runs folder, so ids stay unique across sessions.
    """
    existing: list[int] = [
        int(path.stem.removeprefix("run_"))
        for path in RUNS_DIR.glob("run_*.csv")
        if path.stem.removeprefix("run_").isdigit()
    ]

    return max(existing, default=0) + 1


def monte_carlo(
        num_runs: int,
        num_steps: int,
        dt: float,
        config: dict,
        seed: int | None = None
) -> list[int]:
    """
        Runs independent pursuit-evasion simulations and writes each one to its
        own run_<run_id>.csv file inside the runs folder. The first run uses the
        initial conditions in the config. Every subsequent run draws one distance
        uniformly within the spawn distance bounds and places both aircrafts
        exactly that far from the destination, redrawing until they start at
        least the minimum separation apart. The pursuer's speed is drawn
        uniformly between half and one and a half times the config speed and
        the evader always flies at the evader speed ratio times that speed, both
        in uniformly random directions. Returns the run ids.
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    initial_conditions: dict = config["initial_conditions"]
    parameters: dict = config["simulation"]

    spawn_distance: NDArray[np.float64] = np.array(initial_conditions["spawn_distance"], dtype=np.float64)
    spawn_altitude: NDArray[np.float64] = np.array(initial_conditions["spawn_altitude"], dtype=np.float64)
    min_separation: float = initial_conditions["min_separation"]
    evader_speed_ratio: float = initial_conditions["evader_speed_ratio"]
    intercept_radius: float = parameters["intercept_radius"]
    max_acceleration: float = parameters["max_acceleration"]
    destination: NDArray[np.float64] = np.array(initial_conditions["destination"], dtype=np.float64)

    base_pursuer_position: NDArray[np.float64] = np.array(initial_conditions["pursuer_position"], dtype=np.float64)
    base_pursuer_velocity: NDArray[np.float64] = np.array(initial_conditions["pursuer_velocity"], dtype=np.float64)
    base_evader_position: NDArray[np.float64] = np.array(initial_conditions["evader_position"], dtype=np.float64)
    base_evader_velocity: NDArray[np.float64] = np.array(initial_conditions["evader_velocity"], dtype=np.float64)
    base_speed: float = float(np.linalg.norm(base_pursuer_velocity))

    if not valid_start(
        base_pursuer_position, base_pursuer_velocity,
        base_evader_position, base_evader_velocity,
        destination, min_separation, spawn_distance[0], evader_speed_ratio
    ):
        raise ValueError(
            "The initial conditions in config.yaml must place both aircrafts the same "
            "distance from the destination, at least the lower spawn_distance away from it "
            "and at least min_separation apart, with the evader's speed equal to "
            "evader_speed_ratio times the pursuer's speed."
        )

    if np.any(np.abs(spawn_altitude - destination[2]) > spawn_distance[0]):
        raise ValueError(
            "Every spawn_altitude must lie within the lower spawn_distance of the "
            "destination's altitude, otherwise that distance cannot be reached."
        )

    RUNS_DIR.mkdir(exist_ok=True)

    run_ids: list[int] = []

    for run in range(num_runs):
        if run == 0:
            pursuer_position = base_pursuer_position
            pursuer_velocity = base_pursuer_velocity
            evader_position = base_evader_position
            evader_velocity = base_evader_velocity
        else:
            distance: float = rng.uniform(spawn_distance[0], spawn_distance[1])

            while True:
                pursuer_position = sample_position_at_distance(rng, destination, distance, spawn_altitude)
                evader_position = sample_position_at_distance(rng, destination, distance, spawn_altitude)

                if np.linalg.norm(evader_position - pursuer_position) >= min_separation:
                    break

            pursuer_speed: float = base_speed * rng.uniform(0.5, 1.5)
            pursuer_velocity = sample_direction(rng) * pursuer_speed
            evader_velocity = sample_direction(rng) * pursuer_speed * evader_speed_ratio

        pursuer: Pursuer = Pursuer(
            position=pursuer_position,
            velocity=pursuer_velocity,
            navigation_constant=parameters["navigation_constant"],
            max_acceleration=max_acceleration,
            seeker_half_angle=parameters["seeker_half_angle"]
        )

        evader: Evader = Evader(
            position=evader_position,
            velocity=evader_velocity,
            destination=destination,
            evasion_gain=parameters["evasion_gain"],
            destination_gain=parameters["destination_gain"],
            evasion_range=parameters["evasion_range"],
            max_acceleration=max_acceleration
        )

        run_id: int = next_run_id()

        trajectory = run_simulation(
            pursuer,
            evader,
            dt,
            num_steps,
            intercept_radius
        )

        trajectory.insert(0, "run_id", run_id)
        trajectory.to_csv(RUNS_DIR / f"run_{run_id}.csv", index=False)

        run_ids.append(run_id)

    return run_ids
