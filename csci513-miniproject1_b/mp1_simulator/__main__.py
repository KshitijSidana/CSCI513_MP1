#!/usr/bin/env python

import argparse
import csv
import logging
from collections import deque
from pathlib import Path
from typing import NamedTuple
from matplotlib import pyplot as plt

import numpy as np

from mp1_controller.controller import Controller
from mp1_distance_predictor.predictor import Predictor
from mp1_simulator.simulator import CONFIG, Observation, Simulator

logger = logging.getLogger("SIMULATION")
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s %(name)-12s %(levelname)-8s %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mini-Project 1b: Adaptive Cruise Control in CARLA",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--n-episodes",
        help="Number of simulations to run (defaults to 10)",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--nn-model",
        help="The file where you stored your trained NN model",
        type=lambda p: Path(p).resolve(),
    )

    parser.add_argument(
        "--log-dir",
        help="Directory to store the simulation trace (defaults to 'log/' in the current directory)",
        type=lambda p: Path(p).absolute(),
        default=Path.cwd() / "logs",
    )

    parser.add_argument(
        "--seed", help="Random seed for ado behavior", type=int, default=0
    )

    parser.add_argument(
        "--render", help="Render the Pygame display", action="store_true"
    )

    return parser.parse_args()


class TraceRow(NamedTuple):
    ego_velocity: float
    target_speed: float
    distance_to_lead: float
    ado_velocity: float


def observation_to_trace_row(obs: Observation, sim: Simulator) -> TraceRow:
    row = TraceRow(
        ego_velocity=obs.velocity,
        target_speed=obs.target_velocity,
        distance_to_lead=obs.distance_to_lead,
        ado_velocity=sim._get_ado_velocity(),
    )
    return row


def run_episode(
    sim: Simulator, controller: Controller, predictor: Predictor, *, log_file: Path
):
    trace = deque()  # type: deque[TraceRow]

    obs, img = sim.reset()
    trace.append(observation_to_trace_row(obs, sim))
    while True:
        estimate_dist = predictor.predict(img)
        action = controller.run_step(obs, estimate_dist=estimate_dist)
        obs, img = sim.step(action)

        ## for visualizing the image ##
        # plt.imshow(img, interpolation='nearest')
        # plt.show()

        trace.append(observation_to_trace_row(obs, sim))

        if sim.completed:
            break

    with open(log_file, "w") as flog:
        csv_stream = csv.writer(flog)
        csv_stream.writerow(
            [
                "timestep",
                "time_elapsed",
                "ego_velocity",
                "target_speed",
                "distance_to_lead",
                "lead_speed",
            ]
        )

        for i, row in enumerate(trace):
            row = [
                i,
                sim.dt * i,
                row.ego_velocity,
                row.target_speed,
                row.distance_to_lead,
                row.ado_velocity,
            ]
            csv_stream.writerow(row)


def main():
    args = parse_args()
    n_episodes: int = args.n_episodes
    log_dir: Path = args.log_dir

    if log_dir.is_dir():
        logger.warning(
            "Looks like the log directory %s already exists. Existing logs may be overwritten.",
            str(log_dir),
        )
    else:
        log_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    ado_sawtooth_width = rng.uniform(low=0.2, high=0.8)
    ado_sawtooth_period = rng.uniform(low=5.0, high=15.0)

    sim = Simulator(
        render=args.render,
        ado_sawtooth_period=ado_sawtooth_period,
        ado_sawtooth_width=ado_sawtooth_width,
    )
    controller = Controller(
        distance_threshold=CONFIG["distance_threshold"],
        target_speed=CONFIG["desired_speed"],
    )

    predictor = Predictor(args.nn_model)

    for i in range(n_episodes):
        logger.info("Running Episode %d", i)
        episode_name = "episode-{:05d}.csv".format(i)
        run_episode(sim, controller, predictor, log_file=(log_dir / episode_name))


if __name__ == "__main__":
    main()
