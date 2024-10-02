import os
import pickle
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import neat
from evoman.environment import Environment


class EvoAlgo1:
    def __init__(self, env: Environment, config: Any = None, enemy: int = 1, **kwargs):
        self.env = env
        self.config = config
        self.enemy = enemy
        self.filename = f"neat/enemy_{enemy}"
        self.initialize_directories(self.filename)

    def initialize_directories(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

    def evaluate(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.env.play(pcont=net)[0]  # type: ignore

    def run(self, n_gens=10, run_id: int = 0):
        p = neat.Population(self.config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        winner = p.run(self.evaluate, n_gens)

        # Display the winning genome.
        print("\nBest genome:\n{!s}".format(winner))

        # Show output of the most fit genome against training data.
        print("\nOutput:")
        winner_net = neat.nn.FeedForwardNetwork.create(winner, self.config)

        fitness, _, _, _ = self.env.play(pcont=winner_net)  # type: ignore
        print(f"\nResult: {fitness}")

        return fitness, winner, stats
    
    def plot_stats(
        self, statistics, ylog=False, view=False, filename="avg_fitness.png"
    ):
        """Plots the population's average and best fitness."""

        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        # avg_fitness = np.array(statistics.get_fitness_mean())
        # stdev_fitness = np.array(statistics.get_fitness_stdev())

        # plt.plot(generation, avg_fitness, 'b-', label="average")
        # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
        # plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
        plt.plot(generation, best_fitness, "r-", label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale("symlog")

        plt.savefig(filename)


class EvoAlgo2(EvoAlgo1):
    def __init__(self, env: Environment, config: Any = None, enemy: int = 1, **kwargs):
        super().__init__(env, config, enemy, **kwargs)
        self.beta = kwargs.get("beta", 0.5)
        self.alpha = kwargs.get("alpha", 0.2)

    def evaluate(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            _, plife, elife, playtime = self.env.play(pcont=net)  # type: ignore

            max_life = 100
            fitness = (
                (1 - self.alpha) * (max_life - elife) + self.alpha * plife
            ) ** 2 - self.beta * np.log(playtime)
            fitness = fitness / 100
            
            genome.fitness = fitness
