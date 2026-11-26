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


def offset_direction(
        rng: np.random.Generator,
        objective_direction: NDArray[np.float64],
        offset_angle: float
) -> NDArray[np.float64]:
    """
        Returns a unit vector exactly the offset angle away from the objective
        direction, tilted toward a uniformly random side of it.
    """
    objective: NDArray[np.float64] = objective_direction / np.linalg.norm(objective_direction)

    side: NDArray[np.float64] = rng.normal(size=3)
    side -= np.dot(side, objective) * objective
    side /= np.linalg.norm(side)

    return np.cos(offset_angle) * objective + np.sin(offset_angle) * side


def heading_offset(
        velocity: NDArray[np.float64],
        objective_direction: NDArray[np.float64]
) -> float:
    """
        Returns the angle in radians between an aircraft's velocity and the
        direction of its objective, 0 when it starts flying straight at it.
    """
    cosine: float = np.dot(velocity, objective_direction) / (
        np.linalg.norm(velocity) * np.linalg.norm(objective_direction)
    )

    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def valid_start(
        pursuer_position: NDArray[np.float64],
        pursuer_velocity: NDArray[np.float64],
        evader_position: NDArray[np.float64],
        evader_velocity: NDArray[np.float64],
        destination: NDArray[np.float64],
        min_separation: float,
        min_destination_distance: float,
        evader_speed_ratio: float,
        pursuer_distance_ratio: float,
        max_heading_offset: float
) -> bool:
    """
        Returns True when the start is valid. The pursuer must be exactly the
        pursuer distance ratio times the evader's distance away from the
        destination, so it starts slightly closer and can cut the evader off,
        the pursuer's distance must be at least the minimum destination
        distance, they must start at least the minimum separation apart, and
        the evader must fly at exactly the evader speed ratio times the
        pursuer's speed.

        Both must also start pointed equally well at their objectives, the
        pursuer at the evader and the evader at the destination. Their heading
        offsets must be identical and no larger than the maximum heading offset,
        so neither starts with a turn the other does not have to make.
    """
    pursuer_distance: float = np.linalg.norm(destination - pursuer_position)
    evader_distance: float = np.linalg.norm(destination - evader_position)
    pursuer_speed: float = np.linalg.norm(pursuer_velocity)
    evader_speed: float = np.linalg.norm(evader_velocity)
    pursuer_offset: float = heading_offset(pursuer_velocity, evader_position - pursuer_position)
    evader_offset: float = heading_offset(evader_velocity, destination - evader_position)

    return bool(
        abs(pursuer_distance - pursuer_distance_ratio * evader_distance) <= TOLERANCE * evader_distance
        and pursuer_distance >= min_destination_distance
        and np.linalg.norm(evader_position - pursuer_position) >= min_separation
        and abs(evader_speed - evader_speed_ratio * pursuer_speed) <= TOLERANCE * pursuer_speed
        and abs(pursuer_offset - evader_offset) <= TOLERANCE
        and pursuer_offset <= max_heading_offset + TOLERANCE
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
        uniformly within the spawn distance bounds, places the evader exactly
        that far from the destination and the pursuer exactly the pursuer
        distance ratio times that far, redrawing until they start at least the
        minimum separation apart. The pursuer's speed is drawn
        uniformly between half and one and a half times the config speed and
        the evader always flies at the evader speed ratio times that speed. One
        heading offset is drawn uniformly up to the maximum heading offset and
        both aircrafts start exactly that far off their objectives, the pursuer
        off the evader and the evader off the destination. Returns the run ids.
    """
    rng: np.random.Generator = np.random.default_rng(seed)

    initial_conditions: dict = config["initial_conditions"]
    parameters: dict = config["simulation"]

    spawn_distance: NDArray[np.float64] = np.array(initial_conditions["spawn_distance"], dtype=np.float64)
    spawn_altitude: NDArray[np.float64] = np.array(initial_conditions["spawn_altitude"], dtype=np.float64)
    min_separation: float = initial_conditions["min_separation"]
    evader_speed_ratio: float = initial_conditions["evader_speed_ratio"]
    pursuer_distance_ratio: float = initial_conditions["pursuer_distance_ratio"]
    max_heading_offset: float = np.radians(initial_conditions["max_heading_offset"])
    intercept_radius: float = parameters["intercept_radius"]
    arrival_radius: float = parameters["arrival_radius"]
    destination: NDArray[np.float64] = np.array(initial_conditions["destination"], dtype=np.float64)

    base_pursuer_position: NDArray[np.float64] = np.array(initial_conditions["pursuer_position"], dtype=np.float64)
    base_pursuer_velocity: NDArray[np.float64] = np.array(initial_conditions["pursuer_velocity"], dtype=np.float64)
    base_evader_position: NDArray[np.float64] = np.array(initial_conditions["evader_position"], dtype=np.float64)
    base_evader_velocity: NDArray[np.float64] = np.array(initial_conditions["evader_velocity"], dtype=np.float64)
    base_speed: float = float(np.linalg.norm(base_pursuer_velocity))

    if not valid_start(
        base_pursuer_position, base_pursuer_velocity,
        base_evader_position, base_evader_velocity,
        destination, min_separation, pursuer_distance_ratio * spawn_distance[0],
        evader_speed_ratio, pursuer_distance_ratio, max_heading_offset
    ):
        raise ValueError(
            "The initial conditions in config.yaml must place the pursuer exactly "
            "pursuer_distance_ratio times the evader's distance from the destination, at "
            "least pursuer_distance_ratio times the lower spawn_distance away from it, with "
            "both at least min_separation apart, with the evader's speed equal to "
            "evader_speed_ratio times the pursuer's speed, and with the pursuer's velocity "
            "and the evader's velocity the same angle, at most max_heading_offset, away "
            "from the evader and the destination respectively."
        )

    if np.any(np.abs(spawn_altitude - destination[2]) > pursuer_distance_ratio * spawn_distance[0]):
        raise ValueError(
            "Every spawn_altitude must lie within pursuer_distance_ratio times the lower "
            "spawn_distance of the destination's altitude, otherwise that distance cannot "
            "be reached."
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
                pursuer_position = sample_position_at_distance(
                    rng, destination, pursuer_distance_ratio * distance, spawn_altitude
                )
                evader_position = sample_position_at_distance(rng, destination, distance, spawn_altitude)

                if np.linalg.norm(evader_position - pursuer_position) >= min_separation:
                    break

            pursuer_speed: float = base_speed * rng.uniform(0.5, 1.5)
            offset_angle: float = rng.uniform(0.0, max_heading_offset)

            pursuer_velocity = pursuer_speed * offset_direction(
                rng,
                evader_position - pursuer_position,
                offset_angle
            )
            evader_velocity = pursuer_speed * evader_speed_ratio * offset_direction(
                rng,
                destination - evader_position,
                offset_angle
            )

        pursuer: Pursuer = Pursuer(
            position=pursuer_position,
            velocity=pursuer_velocity,
            navigation_constant=parameters["navigation_constant"],
            max_acceleration=parameters["pursuer_max_acceleration"],
            seeker_half_angle=parameters["seeker_half_angle"]
        )

        evader: Evader = Evader(
            position=evader_position,
            velocity=evader_velocity,
            destination=destination,
            evasion_gain=parameters["evasion_gain"],
            destination_gain=parameters["destination_gain"],
            evasion_range=parameters["evasion_range"],
            max_acceleration=parameters["evader_max_acceleration"]
        )

        run_id: int = next_run_id()

        trajectory = run_simulation(
            pursuer,
            evader,
            dt,
            num_steps,
            intercept_radius,
            arrival_radius
        )

        trajectory.insert(0, "run_id", run_id)
        trajectory.to_csv(RUNS_DIR / f"run_{run_id}.csv", index=False)

        run_ids.append(run_id)

    return run_ids
