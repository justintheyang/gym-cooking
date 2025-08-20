import pygame
import os
import numpy as np
from PIL import Image
from misc.game.game import Game
# from misc.game.utils import *


class GameImage(Game):
    def __init__(self, filename, world, sim_agents, record=False, layout=False, directory=None):
        super().__init__(world, sim_agents)
        self.record = record
        self.layout = layout
        
        record_root = os.environ.get('OVERCOOKED_RECORD_ROOT')
        if directory:
            self.game_record_dir = directory
        elif record_root:
            self.game_record_dir = record_root
        else:
            # fallback to the old location next to this file
            base_dir = os.path.dirname(__file__)
            self.game_record_dir = os.path.join(base_dir, 'records', filename)        


    def on_init(self):
        super().on_init()

        if self.record and not self.layout:
            os.makedirs(self.game_record_dir, exist_ok=True)
            for f in os.listdir(self.game_record_dir):
                path = os.path.join(self.game_record_dir, f)
                if os.path.isfile(path):
                    os.remove(path)

    def get_image_obs(self):
        self.on_render()
        img_int = pygame.PixelArray(self.screen)
        img_rgb = np.zeros([img_int.shape[1], img_int.shape[0], 3], dtype=np.uint8)
        for i in range(img_int.shape[0]):
            for j in range(img_int.shape[1]):
                color = pygame.Color(img_int[i][j])
                img_rgb[j, i, 0] = color.g
                img_rgb[j, i, 1] = color.b
                img_rgb[j, i, 2] = color.r
        return img_rgb

    def save_image_obs(self, t, filename=None):
        if not filename:
            filename = f"t={t:03d}.png"
        if self.record:
            # Ensure directory is still present
            print(f'Saving to {self.game_record_dir}')
            os.makedirs(self.game_record_dir, exist_ok=True)
            frame_path = os.path.join(self.game_record_dir, filename)
            self.on_render()
            pygame.image.save(self.screen, frame_path)
