from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ..simulate.monte_carlo import RUNS_DIR


MAX_FRAMES: int = 400
HOLD_FRAMES: int = 20

CONFIG_PATH: Path = Path(__file__).resolve().parents[3] / "config.yaml"

PURSUER: str = "#D55E00"
EVADER: str = "#0072B2"
DESTINATION: str = "black"

STYLE: dict = {
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
}


def resolve_run_file(name: str) -> Path | None:
    """
        Turns the text typed into the terminal into a csv path inside the runs
        folder, accepting '3', 'run_3' or 'run_3.csv' and returns None when no
        such file exists.
    """
    name = name.strip()

    if name.isdigit():
        name = f"run_{name}"

    if not name.endswith(".csv"):
        name += ".csv"

    path: Path = RUNS_DIR / name

    return path if path.is_file() else None


def read_config() -> dict:
    """
        Returns the contents of config.yaml, or an empty dict when it is missing.
    """
    if not CONFIG_PATH.is_file():
        return {}

    with CONFIG_PATH.open("r") as file:
        return yaml.safe_load(file) or {}


def find_destination(data: pd.DataFrame, config: dict) -> NDArray[np.float64] | None:
    """
        Returns the destination of the run, read from its csv columns when they
        exist and otherwise from config.yaml for runs recorded before the
        destination was saved.
    """
    columns: list[str] = ["destination_x", "destination_y", "destination_z"]

    if all(column in data for column in columns):
        return data[columns].iloc[0].to_numpy(dtype=np.float64)

    destination = config.get("initial_conditions", {}).get("destination")

    return None if destination is None else np.array(destination, dtype=np.float64)


def set_limits(axes, points: NDArray[np.float64]) -> None:
    """
        Fixes the axis limits to the extent of the whole run with a small margin
        and sets the box proportions to match, so that one unit has the same
        length along every axis and the geometry is not distorted.
    """
    margin: float = 0.05 * max(float(np.ptp(points, axis=0).max()), 1.0)
    low: NDArray[np.float64] = points.min(axis=0) - margin
    high: NDArray[np.float64] = points.max(axis=0) + margin

    axes.set_xlim(low[0], high[0])
    axes.set_ylim(low[1], high[1])
    axes.set_zlim(low[2], high[2])
    axes.set_box_aspect(tuple(high - low))


def animate_run(path: Path) -> None:
    """
        Plays a recorded run as a 3D trajectory plot. The pursuer and evader
        paths are drawn up to the current time, with open circles at the
        initial positions, a filled marker at the current positions and a
        black star at the destination. When the run contains an interception
        the evader position at that moment is marked with a cross.
    """
    data: pd.DataFrame = pd.read_csv(path)
    config: dict = read_config()

    pursuer: NDArray[np.float64] = data[["pursuer_x", "pursuer_y", "pursuer_z"]].to_numpy()
    evader: NDArray[np.float64] = data[["evader_x", "evader_y", "evader_z"]].to_numpy()
    time: NDArray[np.float64] = data["time"].to_numpy()
    destination: NDArray[np.float64] | None = find_destination(data, config)
    outcome: str | None = data["outcome"].iloc[0] if "outcome" in data else None

    distance: NDArray[np.float64] = np.linalg.norm(evader - pursuer, axis=1)

    intercept_index: int | None = None

    if "intercepted" in data and data["intercepted"].any():
        intercept_index = int(np.flatnonzero(data["intercepted"].to_numpy())[0])
    elif outcome == "hit":
        intercept_index = len(data) - 1

    last: int = len(data) - 1
    stride: int = max(1, int(np.ceil(len(data) / MAX_FRAMES)))
    frames: list[int] = list(range(0, len(data), stride))

    if frames[-1] != last:
        frames.append(last)

    frames += [last] * HOLD_FRAMES

    dt: float = float(time[1] - time[0]) if len(time) > 1 else 0.1
    interval: float = max(1.0, dt * stride * 1000)

    plt.rcParams.update(STYLE)

    figure = plt.figure(figsize=(8, 6.5))
    axes = figure.add_subplot(111, projection="3d")

    extent: list[NDArray[np.float64]] = [pursuer, evader]

    if destination is not None:
        extent.append(destination.reshape(1, 3))

    set_limits(axes, np.vstack(extent))

    axes.set_xlabel("$x$ (m)")
    axes.set_ylabel("$y$ (m)")
    axes.set_zlabel("$z$ (m)")
    axes.set_title(f"Pursuit-evasion trajectories, {path.stem.replace('_', ' ')}")
    axes.view_init(elev=24, azim=-58)

    pursuer_path, = axes.plot([], [], [], color=PURSUER, linewidth=1.2, label="Pursuer")
    evader_path, = axes.plot([], [], [], color=EVADER, linewidth=1.2, label="Evader")
    pursuer_now, = axes.plot([], [], [], "o", color=PURSUER, markersize=4)
    evader_now, = axes.plot([], [], [], "o", color=EVADER, markersize=4)

    axes.plot(*pursuer[0:1].T, "o", color=PURSUER, markerfacecolor="none", markersize=7, label="Pursuer initial position")
    axes.plot(*evader[0:1].T, "o", color=EVADER, markerfacecolor="none", markersize=7, label="Evader initial position")

    if destination is not None:
        axes.plot(*destination.reshape(3, 1), "*", color=DESTINATION, markersize=9, label="Destination")

    interception, = axes.plot([], [], [], "x", color="black", markersize=9, markeredgewidth=1.5, label="Interception")

    axes.legend(loc="upper left", frameon=False)

    readout = axes.text2D(0.02, 0.02, "", transform=axes.transAxes, fontsize=8)

    def update(frame: int):
        index: int = min(frame, last)

        pursuer_path.set_data_3d(*pursuer[: index + 1].T)
        evader_path.set_data_3d(*evader[: index + 1].T)
        pursuer_now.set_data_3d(*pursuer[index:index + 1].T)
        evader_now.set_data_3d(*evader[index:index + 1].T)

        text: str = f"$t$ = {time[index]:.1f} s,  range = {distance[index]:.1f} m"

        if destination is not None:
            text += f",  evader-destination = {np.linalg.norm(destination - evader[index]):.1f} m"

        if intercept_index is not None and index >= intercept_index:
            point: NDArray[np.float64] = evader[intercept_index]
            interception.set_data_3d([point[0]], [point[1]], [point[2]])
            text += f"  (intercepted at $t$ = {time[intercept_index]:.1f} s)"
        elif index == last and outcome == "miss":
            text += f"  (miss, minimum range {distance.min():.1f} m)"

        readout.set_text(text)

        return pursuer_path, evader_path, pursuer_now, evader_now, interception, readout

    update(0)

    figure._animation = FuncAnimation(
        figure,
        update,
        frames=frames,
        interval=interval,
        blit=False,
        repeat=False
    )

    plt.show()


def ask(question: str) -> bool:
    """
        Asks a yes or no question in the terminal until the answer is one of
        y, yes, n or no, and returns True for yes.
    """
    while True:
        answer: str = input(question).strip().lower()

        if answer in ("y", "yes"):
            return True

        if answer in ("n", "no"):
            return False

        print("Please answer y or n.")


def choose_run() -> Path | None:
    """
        Lists the recorded runs and asks for one until an existing csv file is
        named, returning None when there are no runs to choose from.
    """
    available: list[str] = sorted(
        (path.name for path in RUNS_DIR.glob("run_*.csv")),
        key=lambda name: (len(name), name)
    )

    if not available:
        print(f"No run files found in {RUNS_DIR}")
        return None

    print("Available runs: " + ", ".join(available))

    while True:
        name: str = input("Enter the csv file name (e.g. run_1.csv): ")
        path: Path | None = resolve_run_file(name)

        if path is not None:
            return path

        print(f"'{name.strip()}' was not found in {RUNS_DIR}, try again.")


def main() -> None:
    question: str = "Do you want a visualization of a particular run? (y/n): "

    while ask(question):
        path: Path | None = choose_run()

        if path is None:
            return

        animate_run(path)
        plt.close("all")

        question = "Do you want to visualize another run? (y/n): "
