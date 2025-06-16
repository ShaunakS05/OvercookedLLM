# misc/game/gameplay.py

import os
import pygame
from datetime import datetime
from typing import Tuple
from collections import defaultdict
from multiprocessing.connection import Client

from misc.game.game import Game
from misc.game.utils import KeyToTuple
from utils.interact import interact


class GamePlay(Game):
    def __init__(self, filename, world, sim_agents):
        super().__init__(world, sim_agents, play=True)

        # where to stash screenshots
        self.filename = filename
        self.save_dir = 'misc/game/screenshots'
        os.makedirs(self.save_dir, exist_ok=True)

        # map grid‐square types (optional)
        self.gridsquare_types = defaultdict(set)
        for name, gs_list in self.world.objects.items():
            for gs in gs_list:
                self.gridsquare_types[name].add(gs.location)

        # pipe back to llm process
        self.client = None
        self.current_frame = 0
        self.current_agent_id = 1
        self.current_agent = self.sim_agents[0]
        self.action_str = None
        self.action_loc = None


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
            if self.client:
                self.client.close()

        elif event.type == pygame.KEYDOWN:
            # ENTER → screenshot
            if event.key == pygame.K_RETURN:
                stamp = datetime.now().strftime("%m-%d-%y_%H-%M-%S")
                name = f"{self.filename}_{stamp}.png"
                pygame.image.save(self.screen, os.path.join(self.save_dir, name))
                print(f"[GamePlay] saved screenshot {name}")
                return

            # 1/2/3/4 → switch chef
            name = pygame.key.name(event.key)
            if name in "1234":
                idx = int(name) - 1
                if idx < len(self.sim_agents):
                    self.current_agent_id = idx + 1
                    self.current_agent = self.sim_agents[idx]
                return

            # SPACE → pick up / drop
            if event.key == pygame.K_SPACE:
                self.current_agent.action = (0, 0)
                self.action_str, self.action_loc = interact(self.current_agent, self.world)
                return

            # arrows → move/pose interaction
            if event.key in KeyToTuple:
                self.current_agent.action = KeyToTuple[event.key]
                self.action_str, self.action_loc = interact(self.current_agent, self.world)
                return


    def on_execute(self):
        # initialize pygame & screen
        self.on_init()

        # main loop
        while self._running:
            # send fresh state upstream
            self.__send_state()

            # pump events
            for ev in pygame.event.get():
                self.on_event(ev)

            # redraw
            self.on_render()
            self.current_frame += 1

        # cleanup on exit
        self.on_cleanup()


    def __get_agent_location(self, idx=0) -> Tuple[int, int]:
        assert 0 <= idx < len(self.sim_agents)
        return self.sim_agents[idx].location


    def __send_state(self):
        loc = self.__get_agent_location(self.current_agent_id - 1)
        try:
            if self.client is None:
                self.client = Client(("localhost", 6000))
            self.client.send([
                self.current_frame,
                self.current_agent_id,
                loc,
                self.action_str,
                self.action_loc
            ])
        except Exception as e:
            print(f"[GamePlay] pipe error: {e}")
            self.client = None
