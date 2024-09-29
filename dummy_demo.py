################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import json
import os
from typing import Dict

import random

from deap import base, creator, tools, algorithms
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = "dummy_demo"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


class EvoAlgo1:
    def __init__(self, env: Environment, config: Dict = {}):
        self.env = env
        self.config = config
        self.population = self.init_population()

    def init_population(self):
        # create classes
        creator.create("FitnessMax", base.Fitness, weights=(1,))
        creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

        # create toolbox for evolution
        toolbox = base.Toolbox()
        toolbox.register("attr_float", random.random)
        toolbox.register(
            "individual",
            tools.initRepeat,
            creator.Individual,  # type: ignore
            toolbox.attr_float,  # type: ignore
            n=self.config["n_per_ind"],
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore
        return toolbox.population(n=100)  # type: ignore


# env
if __name__ == "__main__":
    # load network parameters
    params = json.load(open("params.json"))

    # initializes environment with ai player using random controller, playing against static enemy
    print('test')
    env = Environment(experiment_name=experiment_name,
                    enemies=[8],
                    playermode="ai",
                    player_controller=player_controller(params["n_per_ind"]),
                    enemymode="static",
                    level=1,
                    speed="fastest",
                    visuals=False)
    env.play()
    algo = EvoAlgo1(env, config=params)
