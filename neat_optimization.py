################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import os
import pickle
import argparse
import neat
import numpy as np
from demo_controller import player_controller
from evoman.environment import Environment
from neat_algorithms import EvoAlgo1, EvoAlgo2

experiment_name = "neat"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


class neat_controller(player_controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):  # type: ignore
        inputs = (inputs - min(inputs)) / float((max(inputs) - min(inputs)))
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


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", type=str, choices=["ea1", "ea2"], default="ea1")
    parser.add_argument("--enemies", type=int, nargs="+", default=[8])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--n_gens", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    return parser.parse_args()


def eval_genomes(genome, config, experiment_name, enemy, file_path):
    i_gains = []
    for i in range(5):
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=neat_controller(_n_hidden=0),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
        )
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitness, plife, elife, playtime = env.play(pcont=net)  # type: ignore
        i_gains.append(plife - elife)
        print(f"Best genome achieves individual gain for run {i + 1}: {plife - elife}")
    np.save(file_path, np.array(i_gains))


# env
if __name__ == "__main__":
    # load network parameters
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.ini",
    )
    args = argparser()

    algos = {"ea1": EvoAlgo1, "ea2": EvoAlgo2}
    evo_algorithm = algos[args.algorithm]

    enemies = args.enemies
    runs = args.runs
    alpha = args.alpha
    beta = args.beta
    n_gens = args.n_gens

    for enemy in enemies:
        mean_stats = []
        max_stats = []
        best_winner = None
        best_fitness = 0
        filepath = f"{experiment_name}/{args.algorithm}/enemy_{enemy}"
        # check if the directory exists
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        # initializes environment with ai player playing against static enemy
        env = Environment(
            experiment_name=experiment_name,
            enemies=[enemy],
            playermode="ai",
            player_controller=neat_controller(_n_hidden=0),
            enemymode="static",
            level=2,
            speed="fastest",
            visuals=False,
        )
        algo = evo_algorithm(env, config=config, enemy=enemy, alpha=alpha, beta=beta)

        # run the algorithm for the specified number of runs
        for run in range(runs):
            curr_run_best_fitness, curr_run_winner, curr_run_stats = algo.run(
                n_gens=n_gens, run_id=run + 1
            )
            if (curr_run_best_fitness > best_fitness) or (best_winner is None):
                best_winner = curr_run_winner
                best_fitness = curr_run_best_fitness

            mean_stats.append(np.array(curr_run_stats.get_fitness_mean()))
            max_stats.append(
                np.array([c.fitness for c in curr_run_stats.most_fit_genomes])
            )

        # save the best winner genome
        eval_genomes(
            genome=best_winner,
            file_path=f"{filepath}/best_winner_igain.npy",
            config=config,
            experiment_name=experiment_name,
            enemy=enemy,
        )
        with open(f"{filepath}/winner_genome.pkl", "wb") as f:
            pickle.dump(best_winner, f)

        # save the statistics of all runs
        np.save(f"{filepath}/mean_stats.npy", np.array(mean_stats))
        np.save(f"{filepath}/max_stats.npy", np.array(max_stats))
