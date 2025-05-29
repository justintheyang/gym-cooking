# modules for game
from misc.game.game import Game
from misc.game.utils import *
from utils.core import *
from utils.interact import interact

# helpers
import pygame
import numpy as np
import argparse
from collections import defaultdict
from random import randrange
import os
from datetime import datetime
from pathlib import Path

class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        super().__init__(world, sim_agents, play=True)
        self.filename = filename
        # point at the screenshots folder next to this file
        ROOT = Path(__file__).resolve().parents[4]
        self.save_dir = ROOT / 'docs' / 'exp1' / 'assets'
        self.save_dir.mkdir(parents=True, exist_ok=True)
        os.makedirs(self.save_dir, exist_ok=True)

        # tally up all gridsquare types
        self.gridsquares = []
        self.gridsquare_types = defaultdict(set) # {type: set of coordinates of that type}
        for name, gridsquares in self.world.objects.items():
            for gridsquare in gridsquares:
                self.gridsquares.append(gridsquare)
                self.gridsquare_types[name].add(gridsquare.location)


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        elif event.type == pygame.KEYDOWN:
            # Save current image
            if event.key == pygame.K_RETURN:
                image_name = f'{self.filename}.png'
                pygame.image.save(self.screen, '{}/{}'.format(self.save_dir, image_name))
                print('just saved image {} to {}'.format(image_name, self.save_dir))
                return
            
            # Switch current agent
            if pygame.key.name(event.key) in "1234":
                try:
                    self.current_agent = self.sim_agents[int(pygame.key.name(event.key))-1]
                except:
                    pass
                return

            # Control current agent
            x, y = self.current_agent.location
            if event.key in KeyToTuple.keys():
                action = KeyToTuple[event.key]
                self.current_agent.action = action
                interact(self.current_agent, self.world)

    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_render()
        self.on_cleanup()


