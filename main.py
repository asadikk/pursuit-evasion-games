import yaml
from pathlib import Path
import argparse

import sys

sys.path.insert(0, str(Path(__file__).parent / "src"))

from simulator import monte_carlo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Monte Carlo pursuit-evasion simulations.")
    parser.add_argument("--runs", type=int, default=1, help="The amount of independent pursuit-evasion runs that end in either interception/miss")
    parser.add_argument("--duration", type=int, default=40, help="The simulated time in seconds of each pursuit-evasion run")
    parser.add_argument("--hertz", type=int, default=10, help="The time steps per second, suppose hertz=10, then the MC sim would run 10x each second")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed that makes the randomized initial conditions reproducible")

    args = parser.parse_args()

    if args.runs <= 0:
        parser.error("--runs must be greater than 0")

    if args.duration <= 0:
        parser.error("--duration must be greater than 0")

    if args.hertz <= 0:
        parser.error("--hertz must be greater than 0")

    num_runs: int = args.runs
    duration: int = args.duration
    dt: float = 1 / args.hertz
    num_steps: int = int(round(duration / dt))

    config_path: Path = Path(__file__).parent / "config.yaml"

    with config_path.open("r") as file:
        config = yaml.safe_load(file)

    run_ids: list[int] = monte_carlo(num_runs, num_steps, dt, config, args.seed)

    print(f"Wrote {len(run_ids)} run(s) to src/simulator/runs/")


if __name__ == "__main__":
   main()
