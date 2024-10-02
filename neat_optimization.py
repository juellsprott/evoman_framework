################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import os
import argparse
import neat
from demo_controller import player_controller
from evoman.environment import Environment
from neat_algorithms import EvoAlgo1, EvoAlgo2

experiment_name = "neat"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)


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
    

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("algorithm", type=str, choices=["evo1", "evo2"])
    parser.add_argument("--enemies", type=int, nargs="+", default=[8])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--n_gens", type=int, default=10)
    parser.add_argument("--n_hidden", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--beta", type=float, default=0.5)
    return parser.parse_args()
    

# env
if __name__ == "__main__":
    # load network parameters
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, "neat_config.ini")
    args = argparser()
    
    algos = {"evo1": EvoAlgo1, "evo2": EvoAlgo2}
    evo_algorithm = algos[args.algorithm]
    
    enemies = args.enemies
    runs = args.runs
    alpha = args. alpha
    beta = args.beta
    n_gens = args.n_gens
    
    for enemy in enemies:
        for run in range(runs):
            # initializes environment with ai player playing against static enemy
            env = Environment(experiment_name=experiment_name,
                            enemies=enemy,
                            playermode="ai",
                            player_controller=neat_controller(_n_hidden=0),
                            enemymode="static",
                            level=2,
                            speed="fastest",
                            visuals=False)
            algo = evo_algorithm(env, config=config, enemy=enemy, alpha=alpha, beta=beta)
            curr_run_best = algo.run(n_gens=n_gens, run_id=run+1)
            
