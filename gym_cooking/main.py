# from environment import OvercookedEnvironment
# from gym_cooking.envs import OvercookedEnvironment
from recipe_planner.recipe import *
from utils.world import World
from utils.agent import RealAgent, SimAgent, COLORS
from utils.core import *
from misc.game.gameplay import GamePlay
from misc.metrics.metrics_bag import Bag

import numpy as np
import os
import random
import argparse
from collections import namedtuple

import gym


def parse_arguments():
    parser = argparse.ArgumentParser("Overcooked 2 argument parser")

    # Environment
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--num-agents", type=int, required=True)
    parser.add_argument("--max-num-timesteps", type=int, default=100, help="Max number of timesteps to run")
    parser.add_argument("--max-num-subtasks", type=int, default=14, help="Max number of subtasks for recipe")
    parser.add_argument("--seed", type=int, default=1, help="Fix pseudorandom seed")
    parser.add_argument("--with-image-obs", action="store_true", default=False, help="Return observations as images (instead of objects)")

    # Delegation Planner
    parser.add_argument("--beta", type=float, default=1.3, help="Beta for softmax in Bayesian delegation updates")

    # Navigation Planner
    parser.add_argument("--alpha", type=float, default=0.01, help="Alpha for BRTDP")
    parser.add_argument("--tau", type=int, default=2, help="Normalize v diff")
    parser.add_argument("--cap", type=int, default=75, help="Max number of steps in each main loop of BRTDP")
    parser.add_argument("--main-cap", type=int, default=100, help="Max number of main loops in each run of BRTDP")

    # Visualizations
    parser.add_argument("--play", action="store_true", default=False, help="Play interactive game with keys")
    parser.add_argument("--layout", action="store_true", default=False, help="Screenshot empty layout of the level")
    parser.add_argument("--num-start-locations", type=int, choices=[1, 2], default=None,
        help="When used with --layout, outline the start tile(s): 1 = agent1 (blue), 2 = agent1+agent2 (blue/red)")
    parser.add_argument("--record", action="store_true", default=True, help="Save observation at each time step as an image in misc/game/record")

    # Models
    # Valid options: `bd` = Bayes Delegation; `up` = Uniform Priors
    # `dc` = Divide & Conquer; `fb` = Fixed Beliefs; `greedy` = Greedy
    parser.add_argument("--model1", type=str, default=None, help="Model type for agent 1 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model2", type=str, default=None, help="Model type for agent 2 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model3", type=str, default=None, help="Model type for agent 3 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--model4", type=str, default=None, help="Model type for agent 4 (bd, up, dc, fb, or greedy)")
    parser.add_argument("--recipe", type=str, default=None, help="Override recipe name in the level file (e.g., Salad or SaladOL). If level file contains no recipe section, this will be used.")
    
    # Start locations
    parser.add_argument("--start-location-model1", type=str, default=None, help="Start location for agent 1 in format 'x y' (e.g., '3 4')")
    parser.add_argument("--start-location-model2", type=str, default=None, help="Start location for agent 2 in format 'x y' (e.g., '5 6')")

    # Data saving
    parser.add_argument('--output-dir', type=str, default=None,
        help='Where to save metrics pickle')
    parser.add_argument('--output-prefix', type=str, default=None,
        help='Filename prefix for the pickle (no extension)')
    parser.add_argument('--return-timesteps-only', action="store_true", default=False,
        help='Return only timestep count, skip recording/visualization')

    return parser.parse_args()


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def initialize_agents(arglist):
    """
    Always create `num_agents` RealAgents.

    - If the level file has a phase‑2 recipe list, use it.
    - Otherwise, use --recipe if provided, else default to Salad.
    - Do NOT depend on a phase‑3 'start locations' section.
      (Sim agents/positions come from the environment.)
    """
    real_agents = []

    # Resolve the level file path
    if os.path.isfile(arglist.level):
        init_path = arglist.level
    else:
        lvl = arglist.level[:-4] if arglist.level.endswith('.txt') else arglist.level
        base_dir = os.path.dirname(__file__)
        levels_dir = os.path.normpath(os.path.join(base_dir, '..', 'utils', 'levels'))
        init_path = os.path.join(levels_dir, f'{lvl}.txt')

    # Parse recipes (phase 2)
    recipes = []
    phase = 1
    with open(init_path, 'r') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if line == '':
                phase += 1
                continue
            if phase == 2:
                name = line.strip()
                if not name:
                    continue
                try:
                    recipes.append(globals()[name]())
                except KeyError:
                    raise ValueError(f"Unknown recipe '{name}' in {init_path}")

    # Fallback if no recipe section present in file
    if not recipes:
        recipe_name = arglist.recipe if getattr(arglist, "recipe", None) else "Salad"
        try:
            recipes = [globals()[recipe_name]()]
        except KeyError:
            raise ValueError(f"Unknown recipe '{recipe_name}' from --recipe")

    # Create the requested number of RealAgents
    for i in range(arglist.num_agents):
        real_agents.append(
            RealAgent(
                arglist=arglist,
                name=f'agent-{i+1}',
                id_color=COLORS[i],
                recipes=recipes
            )
        )
    return real_agents

def main_loop(arglist):
    """The main loop for running experiments."""
    print("Initializing environment and agents.")
    env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
    obs = env.reset()
    # game = GameVisualize(env)
    real_agents = initialize_agents(arglist=arglist)

    # Info bag for saving pkl files (only if not in timesteps-only mode)
    if not arglist.return_timesteps_only:
        bag = Bag(arglist=arglist, filename=env.filename, directory=env.pickles_dir)
        bag.set_recipe(recipe_subtasks=env.all_subtasks)
    else:
        bag = None

    while not env.done():
        action_dict = {}

        for agent in real_agents:
            action = agent.select_action(obs=obs)
            action_dict[agent.name] = action

        obs, reward, done, info = env.step(action_dict=action_dict)

        # Agents
        for agent in real_agents:
            agent.refresh_subtasks(world=env.world)

        # Saving info (only if not in timesteps-only mode)
        if bag is not None:
            bag.add_status(cur_time=info['t'], real_agents=real_agents)

    # If return-timesteps-only flag is set, just print timesteps and return
    if arglist.return_timesteps_only:
        print(f"TIMESTEPS:{env.t}")
        return

    # Saving final information before saving pkl file (only if not in timesteps-only mode)
    if bag is not None:
        bag.set_collisions(collisions=env.collisions)
        bag.set_termination(termination_info=env.termination_info,
                successful=env.successful)

if __name__ == '__main__':
    arglist = parse_arguments()
    if arglist.play:
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        game = GamePlay(os.path.basename(env.filename), env.world, env.sim_agents)
        game.on_execute()
    elif arglist.layout: 
        env = gym.envs.make("gym_cooking:overcookedEnv-v0", arglist=arglist)
        env.reset()
        filename = os.path.basename(arglist.level).replace('.txt', '.png')
        env.game.save_image_obs(t=0, filename=filename)
    else:
        model_types = [arglist.model1, arglist.model2, arglist.model3, arglist.model4]
        assert len(list(filter(lambda x: x is not None,
            model_types))) == arglist.num_agents, "num_agents should match the number of models specified"
        fix_seed(seed=arglist.seed)
        main_loop(arglist=arglist)


