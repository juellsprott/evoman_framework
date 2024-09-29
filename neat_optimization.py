################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import os
from typing import Any
import neat
import matplotlib.pyplot as plt
import numpy as np

from evoman.environment import Environment
from demo_controller import player_controller

experiment_name = "neat"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


class EvoAlgo1:
    def __init__(self, env: Environment, config: Any = None, enemy: int = 1):
        self.env = env
        self.config = config
        self.enemy = enemy
        self.filename = f"neat-checkpoints/enemy_{enemy}"
        self.initialize_directories(self.filename)
        
    def initialize_directories(self, dir):
        if not os.path.exists(dir):
            os.makedirs(dir)
        
    def evaluate(self, genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = 0
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            genome.fitness = self.env.play(pcont=net)[0] #type: ignore
            
    def run(self, n_gens = 10, run_id: int = 0):
        p = neat.Population(self.config)
        p.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        p.add_reporter(stats)
        p.add_reporter(neat.Checkpointer(10, filename_prefix=f"{self.filename}/run-{run_id}-neat-checkpoint-"))
        winner = p.run(self.evaluate, n_gens)

        # Display the winning genome.
        print('\nBest genome:\n{!s}'.format(winner))

        # Show output of the most fit genome against training data.
        print('\nOutput:')
        winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
        # self.env.visuals = True
        # self.env.speed = "normal"
        result = self.env.play(pcont=winner_net) #type: ignore
        print(f'\nResult: {result}')
        self.plot_stats(stats, view=True, filename=f"{self.filename}/avg_fitness_run{run_id}.svg")

    def plot_stats(self, statistics, ylog=False, view=False, filename='avg_fitness.svg'):
        """ Plots the population's average and best fitness. """


        generation = range(len(statistics.most_fit_genomes))
        best_fitness = [c.fitness for c in statistics.most_fit_genomes]
        # avg_fitness = np.array(statistics.get_fitness_mean())
        # stdev_fitness = np.array(statistics.get_fitness_stdev())

        # plt.plot(generation, avg_fitness, 'b-', label="average")
        # plt.plot(generation, avg_fitness - stdev_fitness, 'g-.', label="-1 sd")
        # plt.plot(generation, avg_fitness + stdev_fitness, 'g-.', label="+1 sd")
        plt.plot(generation, best_fitness, 'r-', label="best")

        plt.title("Population's average and best fitness")
        plt.xlabel("Generations")
        plt.ylabel("Fitness")
        plt.grid()
        plt.legend(loc="best")
        if ylog:
            plt.gca().set_yscale('symlog')

        plt.savefig(filename)

class neat_controller(player_controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller): #type: ignore
        inputs = (inputs-min(inputs))/float((max(inputs)-min(inputs)))
        output = controller.activate(inputs)
		# takes decisions about sprite actions
        if output[0] > 0.5:
            left = 1
        else:
            left = 0
        
        if output[1] > 0.5:
            right = 1
        else:
            right = 0

        if output[2] > 0.5:
            jump = 1
        else:
            jump = 0

        if output[3] > 0.5:
            shoot = 1
        else:
            shoot = 0

        if output[4] > 0.5:
            release = 1
        else:
            release = 0

        return [left, right, jump, shoot, release]
    



# env
if __name__ == "__main__":
    # load network parameters
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini")
    enemies = [1, 5, 8]
    runs = 1
    for enemy in enemies:
        for run in range(runs):
            # initializes environment with ai player playing against static enemy
            env = Environment(experiment_name=experiment_name,
                            enemies=[enemy],
                            playermode="ai",
                            player_controller=neat_controller(_n_hidden=0),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
            algo = EvoAlgo1(env, config=config, enemy=enemy)
            algo.run(n_gens=20, run_id=run+1)
