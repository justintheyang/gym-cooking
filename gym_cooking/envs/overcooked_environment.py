# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe
from recipe_planner.recipe import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planning
import navigation_planner.utils as nav_utils

# Other core modules
from utils.interact import interact
from utils.world import World
from utils.core import *
from utils.agent import SimAgent
from misc.game.gameimage import GameImage
from utils.agent import COLORS

import copy
import networkx as nx
import numpy as np
import os
import pygame
from itertools import combinations, permutations, product
from collections import namedtuple

import gym
from gym import error, spaces, utils
from gym.utils import seeding


CollisionRepr = namedtuple("CollisionRepr", "time agent_names agent_locations")


class OvercookedEnvironment(gym.Env):
    """Environment object for Overcooked."""

    def __init__(self, arglist):
        self.arglist = arglist
        self.t = 0
        self.set_filename()
        self.set_directory()

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

    def get_repr(self):
        return self.world.get_repr() + tuple([agent.get_repr() for agent in self.sim_agents])

    def __str__(self):
        # Print the world and agents.
        _display = list(map(lambda x: ''.join(map(lambda y: y + ' ', x)), self.rep))
        return '\n'.join(_display)

    def __eq__(self, other):
        return self.get_repr() == other.get_repr()

    def __copy__(self):
        new_env = OvercookedEnvironment(self.arglist)
        new_env.__dict__ = self.__dict__.copy()
        new_env.world = copy.copy(self.world)
        new_env.sim_agents = [copy.copy(a) for a in self.sim_agents]
        new_env.distances = self.distances

        # Make sure new objects and new agents' holdings have the right pointers.
        for a in new_env.sim_agents:
            if a.holding is not None:
                a.holding = new_env.world.get_object_at(
                        location=a.location,
                        desired_obj=None,
                        find_held_objects=True)
        return new_env

    def set_filename(self):
        if self.arglist.output_prefix is not None:
            self.filename = f'{self.arglist.output_prefix}-seed={self.arglist.seed}'
        elif self.arglist.play:
            self.filename = self.arglist.level
        else:
            self.filename = "{}_agents{}_seed{}".format(self.arglist.level,\
                self.arglist.num_agents, self.arglist.seed)
            model = ""
            if self.arglist.model1 is not None:
                model += "_model1-{}".format(self.arglist.model1)
            if self.arglist.model2 is not None:
                model += "_model2-{}".format(self.arglist.model2)
            if self.arglist.model3 is not None:
                model += "_model3-{}".format(self.arglist.model3)
            if self.arglist.model4 is not None:
                model += "_model4-{}".format(self.arglist.model4)
            self.filename += model
    
    def set_directory(self):
        if self.arglist.output_dir is not None:
            self.directory = self.arglist.output_dir + os.sep
            self.pickles_dir = os.path.join(self.directory, 'pickles')
        elif self.arglist.play:
            self.directory = "data/play/"
        else:
            self.directory = "misc/metrics/pickles/"
        os.makedirs(self.directory, exist_ok=True)

    def load_level(self, level, num_agents):
        if os.path.isfile(level):
            level_path = level
        else:
            # strip any trailing “.txt”
            lvl = level[:-4] if level.endswith('.txt') else level
            # locate the built-in levels folder
            levels_dir = os.path.normpath(
                os.path.join(os.path.dirname(__file__),
                             '..', 'utils', 'levels'))
            level_path = os.path.join(levels_dir, f'{lvl}.txt')

        x = 0
        y = 0
        found_recipe_section = False
        found_starts_section = False
        with open(level_path, 'r') as file:
            # Mark the phases of reading.
            phase = 1
            for line in file:
                line = line.strip('\n')
                if line == '':
                    phase += 1
                    continue

                # Phase 1: Read in kitchen map.
                if phase == 1:
                    for x, rep in enumerate(line):
                        # Object, i.e. Tomato, Lettuce, Onion, or Plate.
                        if rep in 'tlop':
                            counter = Counter(location=(x, y))
                            obj = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep]())
                            counter.acquire(obj=obj)
                            self.world.insert(obj=counter)
                            self.world.insert(obj=obj)
                        # Food dispenser: make food dispenser object and food on top as a hack
                        elif rep in 'TLOP':
                            obj = RepToClass[rep]((x, y))
                            # self.world.objects.setdefault(obj.name, []).append(obj)
                            obj2 = Object(
                                    location=(x, y),
                                    contents=RepToClass[rep.lower()]())
                            obj.acquire(obj=obj2)
                            self.world.insert(obj=obj)
                            self.world.insert(obj=obj2)
                        # GridSquare, i.e. Floor, Counter, Cutboard, Delivery.
                        elif rep in RepToClass:
                            newobj = RepToClass[rep]((x, y))
                            self.world.objects.setdefault(newobj.name, []).append(newobj)
                        else:
                            # Empty. Set a Floor tile.
                            f = Floor(location=(x, y))
                            self.world.objects.setdefault('Floor', []).append(f)
                    y += 1
                # Phase 2: Read in recipe list.
                if phase == 2:
                    found_recipe_section = True
                    # tolerate accidental empties handled above
                    self.recipes.append(globals()[line]())
                    continue

                # Phase 3: Read in agent locations (up to num_agents).
                if phase == 3:
                    found_starts_section = True
                    loc = line.split(' ')
                    start_xy = (int(loc[0]), int(loc[1]))
                    self.world.start_locations.append(start_xy)
                    if len(self.sim_agents) < num_agents:
                        sim_agent = SimAgent(
                                name='agent-'+str(len(self.sim_agents)+1),
                                id_color=COLORS[len(self.sim_agents)],
                                location=start_xy)
                        self.sim_agents.append(sim_agent)

        self.distances = {}
        self.world.width = x+1
        self.world.height = y
        self.world.perimeter = 2*(self.world.width + self.world.height)
        
        # --- If no recipe section, use override or default ---
        if not found_recipe_section:
            # Prefer CLI override; else default to Salad
            recipe_name = self.arglist.recipe if hasattr(self.arglist, "recipe") and self.arglist.recipe else "Salad"
            self.recipes.append(globals()[recipe_name]())

        # --- If no start-locations section, use provided start locations or auto-place agents ---
        if not found_starts_section:
            # Check if start locations were provided via command line arguments
            start_locations = [self.arglist.start_location_model1, self.arglist.start_location_model2]
            provided_start_locations = list(filter(lambda x: x is not None, start_locations))
            
            # Validate that number of provided start locations matches num_agents
            if provided_start_locations:
                assert len(provided_start_locations) == num_agents, \
                    f"Number of provided start locations ({len(provided_start_locations)}) should match num_agents ({num_agents})"
            
            if provided_start_locations:
                # Use provided start locations
                for i, start_loc_str in enumerate(provided_start_locations):
                    loc = start_loc_str.split(' ')
                    start_xy = (int(loc[0]), int(loc[1]))
                    self.world.start_locations.append(start_xy)
                    sim_agent = SimAgent(
                            name='agent-'+str(len(self.sim_agents)+1),
                            id_color=COLORS[len(self.sim_agents)],
                            location=start_xy)
                    self.sim_agents.append(sim_agent)
            else:
                # Fall back to auto-placement
                self.auto_place_agents(num_agents)


    def reset(self):
        self.world = World(arglist=self.arglist)
        self.recipes = []
        self.sim_agents = []
        self.world.start_locations = []
        self.agent_actions = {}
        self.t = 0

        # For visualizing episode.
        self.rep = []

        # For tracking data during an episode.
        self.collisions = []
        self.termination_info = ""
        self.successful = False

        # Load world & distances.
        self.load_level(
                level=self.arglist.level,
                num_agents=self.arglist.num_agents)
        self.all_subtasks = self.run_recipes()
        self.world.make_loc_to_gridsquare()
        self.world.make_reachability_graph()
        self.cache_distances()
        self.obs_tm1 = copy.copy(self)

        if self.arglist.record or self.arglist.with_image_obs:
            if self.arglist.record and self.arglist.layout:
                path = self.arglist.output_prefix
            else: 
                path = os.path.join(
                        self.directory, 
                        'records',
                        self.arglist.output_prefix, 
                        f'seed={self.arglist.seed}')
            self.game = GameImage(
                    filename=self.filename,
                    world=self.world,
                    sim_agents=self.sim_agents,
                    record=self.arglist.record,
                    layout=self.arglist.layout,
                    directory=path)
            
            self.game.on_init()
            if self.arglist.record and not self.arglist.layout and not getattr(self.arglist, 'return_timesteps_only', False):
                self.game.save_image_obs(self.t)
            elif self.arglist.record and self.arglist.layout and not getattr(self.arglist, 'return_timesteps_only', False):
                level = os.path.splitext(os.path.basename(self.arglist.level))[0]
                layout_path = os.path.join(
                    self.game.game_record_dir, 
                    f'agents={self.arglist.num_start_locations}',
                    f'{level}.png')
                
                os.makedirs(os.path.dirname(layout_path), exist_ok=True)

                self.game.on_render()
                pygame.image.save(self.game.screen, layout_path)

        return copy.copy(self)

    def close(self):
        return

    def step(self, action_dict):
        # Track internal environment info.
        self.t += 1
        print("===============================")
        print("[environment.step] @ TIMESTEP {}".format(self.t))
        print("===============================")

        # Get actions.
        for sim_agent in self.sim_agents:
            sim_agent.action = action_dict[sim_agent.name]

        # Check collisions.
        self.check_collisions()
        self.obs_tm1 = copy.copy(self)

        # Execute.
        self.execute_navigation()

        # Visualize.
        self.display()
        self.print_agents()
        if self.arglist.record and not getattr(self.arglist, 'return_timesteps_only', False):
            self.game.save_image_obs(self.t)

        # Get a plan-representation observation.
        new_obs = copy.copy(self)
        # Get an image observation
        image_obs = self.game.get_image_obs()

        done = self.done()
        reward = self.reward()
        info = {"t": self.t, "obs": new_obs,
                "image_obs": image_obs,
                "done": done, "termination_info": self.termination_info}
        return new_obs, reward, done, info


    def done(self):
        # Done if the episode maxes out
        if self.t >= self.arglist.max_num_timesteps and self.arglist.max_num_timesteps:
            self.termination_info = "Terminating because passed {} timesteps".format(
                    self.arglist.max_num_timesteps)
            self.successful = False
            return True

        assert any([isinstance(subtask, recipe.Deliver) for subtask in self.all_subtasks]), "no delivery subtask"

        # Done if subtask is completed.
        for subtask in self.all_subtasks:
            # Double check all goal_objs are at Delivery.
            if isinstance(subtask, recipe.Deliver):
                _, goal_obj = nav_utils.get_subtask_obj(subtask)

                delivery_loc = list(filter(lambda o: o.name=='Delivery', self.world.get_object_list()))[0].location
                goal_obj_locs = self.world.get_all_object_locs(obj=goal_obj)
                if not any([gol == delivery_loc for gol in goal_obj_locs]):
                    self.termination_info = ""
                    self.successful = False
                    return False

        self.termination_info = "Terminating because all deliveries were completed"
        self.successful = True
        return True

    def reward(self):
        return 1 if self.successful else 0

    def print_agents(self):
        for sim_agent in self.sim_agents:
            sim_agent.print_status()

    def display(self):
        self.update_display()
        print(str(self))

    def update_display(self):
        self.rep = self.world.update_display()
        for agent in self.sim_agents:
            x, y = agent.location
            self.rep[y][x] = str(agent)


    def get_agent_names(self):
        return [agent.name for agent in self.sim_agents]

    def run_recipes(self):
        """Returns different permutations of completing recipes."""
        self.sw = STRIPSWorld(world=self.world, recipes=self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]
        print('Subtasks:', all_subtasks, '\n')
        return all_subtasks

    def get_AB_locs_given_objs(self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Returns list of locations relevant for subtask's Merge operator.

        See Merge operator formalism in our paper, under Fig. 11:
        https://arxiv.org/pdf/2003.11778.pdf"""

        # For Merge operator on Chop subtasks, we look at objects that can be
        # chopped and the cutting board objects.
        if isinstance(subtask, recipe.Chop):
            # A: Object that can be chopped.
            A_locs = self.world.get_object_locs(obj=start_obj, is_held=False) + list(map(lambda a: a.location,\
                list(filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))

            # B: Cutboard objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

        # For Merge operator on Deliver subtasks, we look at objects that can be
        # delivered and the Delivery object.
        elif isinstance(subtask, recipe.Deliver):
            # B: Delivery objects.
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)

            # A: Object that can be delivered.
            A_locs = self.world.get_object_locs(
                    obj=start_obj, is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj, self.sim_agents))))
            A_locs = list(filter(lambda a: a not in B_locs, A_locs))

        # For Merge operator on Merge subtasks, we look at objects that can be
        # combined together. These objects are all ingredient objects (e.g. Tomato, Lettuce).
        elif isinstance(subtask, recipe.Merge):
            A_locs = self.world.get_object_locs(
                    obj=start_obj[0], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[0], self.sim_agents))))
            B_locs = self.world.get_object_locs(
                    obj=start_obj[1], is_held=False) + list(
                            map(lambda a: a.location, list(
                                filter(lambda a: a.name in subtask_agent_names and a.holding == start_obj[1], self.sim_agents))))

        # For Get(X), you need to go from your current location(s) (A) to a dispenser (B).
        # A: current locations of the subtask’s agent(s)
        elif isinstance(subtask, recipe.Get):
            ## TODO: ATTEMPT -- remove A_locs
            A_locs = [a.location for a in self.sim_agents if a.name in subtask_agent_names]
            # B: all dispenser tiles for that ingredient (already encoded as a “subtask action obj”)
            B_locs = self.world.get_all_object_locs(obj=subtask_action_obj)
            return A_locs, B_locs

        else:
            return [], []

        return A_locs, B_locs

    def get_lower_bound_for_subtask_given_objs(
            self, subtask, subtask_agent_names, start_obj, goal_obj, subtask_action_obj):
        """Return the lower bound distance (shortest path) under this subtask between objects."""
        assert len(subtask_agent_names) <= 2, 'passed in {} agents but can only do 1 or 2'.format(len(agents))

        # Calculate extra holding penalty if the object is irrelevant.
        holding_penalty = 0.0
        for agent in self.sim_agents:
            if agent.name in subtask_agent_names:
                # Check for whether the agent is holding something.
                if agent.holding is not None:
                    if isinstance(subtask, recipe.Merge):
                        continue
                    else:
                        if agent.holding != start_obj and agent.holding != goal_obj:
                            # Add one "distance"-unit cost
                            holding_penalty += 1.0
        # Account for two-agents where we DON'T want to overpenalize.
        holding_penalty = min(holding_penalty, 1)

        # Get current agent locations.
        agent_locs = [agent.location for agent in list(filter(lambda a: a.name in subtask_agent_names, self.sim_agents))]
        A_locs, B_locs = self.get_AB_locs_given_objs(
                subtask=subtask,
                subtask_agent_names=subtask_agent_names,
                start_obj=start_obj,
                goal_obj=goal_obj,
                subtask_action_obj=subtask_action_obj)

        # Add together distance and holding_penalty.
        return self.world.get_lower_bound_between(
                subtask=subtask,
                agent_locs=tuple(agent_locs),
                A_locs=tuple(A_locs),
                B_locs=tuple(B_locs)) + holding_penalty

    def is_collision(self, agent1_loc, agent2_loc, agent1_action, agent2_action):
        """Returns whether agents are colliding.

        Collisions happens if agent collide amongst themselves or with world objects."""
        # Tracks whether agents can execute their action.
        execute = [True, True]

        # Collision between agents and world objects.
        agent1_next_loc = tuple(np.asarray(agent1_loc) + np.asarray(agent1_action))
        if self.world.get_gridsquare_at(location=agent1_next_loc).collidable:
            # Revert back because agent collided.
            agent1_next_loc = agent1_loc

        agent2_next_loc = tuple(np.asarray(agent2_loc) + np.asarray(agent2_action))
        if self.world.get_gridsquare_at(location=agent2_next_loc).collidable:
            # Revert back because agent collided.
            agent2_next_loc = agent2_loc

        # Inter-agent collision.
        if agent1_next_loc == agent2_next_loc:
            if agent1_next_loc == agent1_loc and agent1_action != (0, 0):
                execute[1] = False
            elif agent2_next_loc == agent2_loc and agent2_action != (0, 0):
                execute[0] = False
            else:
                execute[0] = False
                execute[1] = False

        # Prevent agents from swapping places.
        elif ((agent1_loc == agent2_next_loc) and
                (agent2_loc == agent1_next_loc)):
            execute[0] = False
            execute[1] = False
        return execute

    def check_collisions(self):
        """Checks for collisions and corrects agents' executable actions.

        Collisions can either happen amongst agents or between agents and world objects."""
        execute = [True for _ in self.sim_agents]

        # Check each pairwise collision between agents.
        for i, j in combinations(range(len(self.sim_agents)), 2):
            agent_i, agent_j = self.sim_agents[i], self.sim_agents[j]
            exec_ = self.is_collision(
                    agent1_loc=agent_i.location,
                    agent2_loc=agent_j.location,
                    agent1_action=agent_i.action,
                    agent2_action=agent_j.action)

            # Update exec array and set path to do nothing.
            if not exec_[0]:
                execute[i] = False
            if not exec_[1]:
                execute[j] = False

            # Track collisions.
            if not all(exec_):
                collision = CollisionRepr(
                        time=self.t,
                        agent_names=[agent_i.name, agent_j.name],
                        agent_locations=[agent_i.location, agent_j.location])
                self.collisions.append(collision)

        print('\nexecute array is:', execute)

        # Update agents' actions if collision was detected.
        for i, agent in enumerate(self.sim_agents):
            if not execute[i]:
                agent.action = (0, 0)
            print("{} has action {}".format(color(agent.name, agent.color), agent.action))

    def execute_navigation(self):
        for agent in self.sim_agents:
            interact(agent=agent, world=self.world)
            self.agent_actions[agent.name] = agent.action


    def cache_distances(self):
        """Saving distances between world objects."""
        counter_grid_names = [name for name in self.world.objects if "Supply" in name or "Counter" in name or "Delivery" in name or "Cut" in name]
        # Getting all source objects.
        source_objs = copy.copy(self.world.objects["Floor"])
        for name in counter_grid_names:
            source_objs += copy.copy(self.world.objects[name])
        # Getting all destination objects.
        dest_objs = source_objs

        # From every source (Counter and Floor objects),
        # calculate distance to other nodes.
        for source in source_objs:
            self.distances[source.location] = {}
            # Source to source distance is 0.
            self.distances[source.location][source.location] = 0
            for destination in dest_objs:
                # Possible edges to approach source and destination.
                source_edges = [(0, 0)] if not source.collidable else World.NAV_ACTIONS
                destination_edges = [(0, 0)] if not destination.collidable else World.NAV_ACTIONS
                # Maintain shortest distance.
                shortest_dist = np.inf
                for source_edge, dest_edge in product(source_edges, destination_edges):
                    try:
                        dist = nx.shortest_path_length(self.world.reachability_graph, (source.location,source_edge), (destination.location, dest_edge))
                        # Update shortest distance.
                        if dist < shortest_dist:
                            shortest_dist = dist
                    except:
                        continue
                # Cache distance floor -> counter.
                self.distances[source.location][destination.location] = shortest_dist

        # Save all distances under world as well.
        self.world.distances = self.distances

    def is_open_start_tile(self, xy):
        """Open if it's inside the grid, not collidable, and not a supply/cut/delivery/counter."""
        x, y = xy
        if x < 0 or y < 0 or x >= self.world.width or y >= self.world.height:
            return False
        gs = self.world.get_gridsquare_at(location=xy)
        # Start tiles must be non-collidable (Floor) and not occupied by a stationary collidable
        return (gs is not None) and (not gs.collidable)

    def scan_open(self, start_xy, dir_primary, dir_secondary):
        """
        Scan within the interior bounding box (exclude outer walls) in the pattern you specified:
        - Agent 1: move left along the row; when exhausted, move down a row, reset x to right interior.
        - Agent 2: move up along the column; when exhausted, move right a column, reset y to bottom interior.
        """
        # interior bounds (exclude border walls)
        xmin, xmax = 1, max(1, self.world.width - 2)
        ymin, ymax = 1, max(1, self.world.height - 2)

        # clamp start into interior
        x, y = start_xy
        x = min(max(x, xmin), xmax)
        y = min(max(y, ymin), ymax)

        visited = set()
        for _ in range(max(1, (xmax - xmin + 1) * (ymax - ymin + 1))):
            if (x, y) not in visited and self.is_open_start_tile((x, y)):
                return (x, y)
            visited.add((x, y))

            # step primary
            x += dir_primary[0]
            y += dir_primary[1]

            # if moved out of interior, wrap & step secondary
            if not (xmin <= x <= xmax and ymin <= y <= ymax):
                # undo last step
                x -= dir_primary[0]
                y -= dir_primary[1]
                # step secondary
                x += dir_secondary[0]
                y += dir_secondary[1]
                # reset along primary axis to appropriate interior edge
                if dir_primary[0] != 0:  # horizontal primary (agent 1)
                    x = xmax if dir_primary[0] < 0 else xmin
                if dir_primary[1] != 0:  # vertical primary (agent 2)
                    y = ymax if dir_primary[1] < 0 else ymin
                # clamp
                x = min(max(x, xmin), xmax)
                y = min(max(y, ymin), ymax)
        # As a last resort, return top-left interior
        return (xmin, ymin)

    def auto_place_agents(self, num_agents):
        """
        Places agents by your policy:
        - agent1: start at top-right interior (width-2, 1); scan left, then down rows
        - agent2: start at bottom-left interior (1, height-2); scan up, then right columns
        - agent3/4: nominal (6,6) but we never use them; placeholders.
        """
        w, h = self.world.width, self.world.height
        top_right_interior = (max(1, w - 2), 1)
        bottom_left_interior = (1, max(1, h - 2))

        starts = []
        # Agent 1
        a1 = self.scan_open(top_right_interior, dir_primary=(-1, 0), dir_secondary=(0, 1))
        starts.append(a1)
        # Agent 2
        a2 = self.scan_open(bottom_left_interior, dir_primary=(0, -1), dir_secondary=(1, 0))
        if a2 == a1:
            a2 = self.scan_open((a2[0], a2[1]-1), dir_primary=(0, -1), dir_secondary=(1, 0))
        starts.append(a2)

        # Agents 3 & 4: placeholders
        a34 = (6, 6)
        starts.append(a34)
        starts.append(a34)

        # Register starts and create SimAgents (up to num_agents)
        for idx, xy in enumerate(starts[:max(2, num_agents)]):
            self.world.start_locations.append(xy)
            if len(self.sim_agents) < num_agents:
                sim_agent = SimAgent(
                    name=f'agent-{len(self.sim_agents)+1}',
                    id_color=COLORS[len(self.sim_agents)],
                    location=xy
                )
                self.sim_agents.append(sim_agent)
