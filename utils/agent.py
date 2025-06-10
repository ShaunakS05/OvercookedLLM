# (C) Yoshi Sato <satyoshi.com>

# Recipe planning
from recipe_planner.stripsworld import STRIPSWorld
import recipe_planner.utils as recipe_utils
from recipe_planner.utils import *

# Delegation planning
from delegation_planner.bayesian_delegator import BayesianDelegator

# Navigation planner
from navigation_planner.planners.e2e_brtdp import E2E_BRTDP
import navigation_planner.utils as nav_utils

# Other core modules
from utils.core import Counter, Cutboard
from utils.utils import agent_settings

import numpy as np
import copy
from termcolor import colored as color
from collections import namedtuple
import time
from multiprocessing.connection import Listener
from typing import List, Tuple
from copy import deepcopy
import json
import sys
import copy
import os
from pynput.keyboard import Key, Controller

from utils.astar import *
from utils.environment import *
from utils.utils import *
from utils.chatgpt import ChatGPT, g_openai_config
import pygame



AgentRepr = namedtuple("AgentRepr", "name location holding")

# Colors for agents.
COLORS = ['blue', 'magenta', 'yellow', 'green']


g_keyboard = Controller()
class LLMAgent:
    def __init__(self, id: int, arglist):
        assert 0 <= id <= 4
        self.id = id
        self.location = None
        self.on_hand = None
        self.level = None
        self.item_locations = ITEM_LOCATIONS
        self.history = []
        self.prev_state = None
        global g_max_steps
        if arglist.level == "open-divider_salad":
            self.level = OPEN_DIVIDER_SALAD
        elif arglist.level == "open-divider_salad_large":
            self.level = OPEN_DIVIDER_SALAD_L
            self.item_locations = ITEM_LOCATIONS_L
            g_max_steps = 200
        elif arglist.level == "partial-divider_salad":
            self.level = PARTIAL_DEVIDER_SALAD
        elif arglist.level == "partial-divider_salad_large":
            self.level = PARTIAL_DEVIDER_SALAD_L
            self.item_locations = ITEM_LOCATIONS_L
            g_max_steps = 200
        elif arglist.level == "full-divider_salad":
            self.level = FULL_DIVIDER_SALAD
        else:
            assert False, f"unknown level: {arglist.level}"

    def set_state(self, location: Tuple[int, int], action_str: str, action_loc: Tuple[int, int]):
        """ set the latest game state
        Args:
            location (Tuple[int, int]): agent's current location
            action_str (str): action taken by the agent
            action_loc (Tuple[int, int]): location where the action was taken
        """
        self.location = location
        if action_str is None:
            return
        if self.prev_state is not None:
            # discard duplicate state
            if (self.prev_state[0] == location) and (self.prev_state[1] == action_str) and (self.prev_state[2] == action_loc):
                return
        description = action_str
        items: List[str] = identify_items_at(action_loc, self.item_locations)
        if len(items) > 0:
            # remove duplicated items
            if ("sliced" in description) or ("picked" in description):
                if "tomato" in description:
                    items.remove("tomato")
                if "lettuce" in description:
                    items.remove("lettuce")
                if ("picked" in description) and (len(items) > 0):
                    description += " from"
            # change description for merged plate
            elif ("merged plate" in description) and (self.on_hand is not None):
                description = "put sliced " + ", ".join(self.on_hand) + " onto"
            description += ' ' + ", ".join(items)
            print(colors.GREEN + f"agent{self.id}.set_state(): " + description + colors.ENDC)
        self.history.append(description)
        if "picked" in description:
            # identify what item was picked up
            for item in self.item_locations.keys():
                if (item in description) and (item in MOVABLES):
                    if self.on_hand is None:
                        self.on_hand = [item]
                    else:
                        self.on_hand.append(item)
        elif ("put" in description) or ("merged" in description):
            if self.on_hand is not None:
                # update the location of the item
                for item in MOVABLES:
                    for obj in self.on_hand:
                        if item in obj:
                            self.item_locations[item] = action_loc
                self.on_hand = None
        if self.on_hand is not None:
            print(colors.YELLOW + f"agent{self.id}.on_hand = {self.on_hand}" + colors.ENDC)
        self.prev_state = (location, action_str, action_loc)

    def reset_state(self, reset_on_hand: bool=False):
        """ reset the game state of the agent
        Args:
            reset_on_hand (bool, optional): reset the on_hand variable. Defaults to False.
        """
        self.location = None
        self.action_str = None
        self.action_loc = None
        if reset_on_hand:
            self.on_hand = None

    def plan_move_towards(self, dst: Tuple[int, int]) -> Tuple[int, int]:
        """
        First try an A* route to `dst`.  If that point is blocked,
        try each 4-neighbour square around it.  As a last resort fall
        back to the greedy step.
        """
        if self.location == dst:
            return (0, 0)

        def try_astar(target: Tuple[int, int]):
            path = find_path(self.location, target, self.level, verbose=False)
            return path if path and len(path) >= 2 else None

        # 1️⃣  straight to the goal
        path = try_astar(dst)

        # 2️⃣  if the dispenser itself is blocked, aim for a touching tile
        if path is None and self.level[dst[0]][dst[1]]:          # non-walkable
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nbr = (dst[0] + dx, dst[1] + dy)
                # skip if neighbour is outside map or blocked
                if not (0 <= nbr[0] < len(self.level) and
                        0 <= nbr[1] < len(self.level[0]) and
                        self.level[nbr[0]][nbr[1]] == 0):
                    continue
                path = try_astar(nbr)
                if path:
                    break

        # 3️⃣  good A* route found
        if path:
            next_sq = path[1]
            return (next_sq[0] - self.location[0],
                    next_sq[1] - self.location[1])

        # 4️⃣  still no path → fall back to greedy single-step
        dx, dy = dst[0] - self.location[0], dst[1] - self.location[1]
        if abs(dx) > abs(dy):
            return (-1, 0) if dx < 0 else (1, 0)
        else:
            return (0, -1) if dy < 0 else (0, 1)

    def fetch(self, item: str) -> bool:
        # 1. if already in hand, done
        if self.on_hand and item in self.on_hand:
            return True

        # 2. compute next step
        dst, _ = get_dst_tuple(item, self.level, self.item_locations)
        step = self.plan_move_towards(dst)

        # 3. post into pygame event queue
        key = {
            (-1,0): pygame.K_LEFT,
            ( 1,0): pygame.K_RIGHT,
            ( 0,-1): pygame.K_UP,
            ( 0, 1): pygame.K_DOWN
        }.get(step, pygame.K_SPACE)  # SPACE when step==(0,0)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': key}))
        pygame.event.post(pygame.event.Event(pygame.KEYUP,   {'key': key}))

        time.sleep(0.12)

        # 4. return True once set_state() has updated on_hand
        return bool(self.on_hand and item in self.on_hand)

    def put_onto(self, item: str) -> bool:
        if not self.on_hand:
            return True

        dst, _ = get_dst_tuple(item, self.level, self.item_locations)
        step = self.plan_move_towards(dst)

        key = {
            (-1,0): pygame.K_LEFT,
            ( 1,0): pygame.K_RIGHT,
            ( 0,-1): pygame.K_UP,
            ( 0, 1): pygame.K_DOWN
        }.get(step, pygame.K_SPACE)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': key}))
        pygame.event.post(pygame.event.Event(pygame.KEYUP,   {'key': key}))

        time.sleep(0.12)
        return not bool(self.on_hand)

    def slice_on(self, item: str) -> bool:
        """ slice food at the specified item's location
        Args:
            item (str): the name of the item to chop on (must be a cutboard)
        Returns:
            bool: True if the task is closed
        """
        if not(item in self.item_locations.keys()):
            print(colors.RED + f"agent{self.id}.slice_on(): invalid item: {item}" + colors.ENDC)
            return True
        if not("cutboard" in item):
            print(colors.RED + f"agent{self.id}.slice_on(): cannot slice on {item}" + colors.ENDC)
            return True
        destination: Tuple[int, int] = self.item_locations[item]
        for description in self.history[::-1]:
            if ("put" in description) and (item in description):
                self.move_to(destination)
                break
            elif "sliced" in description:
                return True
        return False

    def deliver(self, dummy=None) -> bool:
        dst = list(self.item_locations["star"]); dst[0] += 1
        step = self.plan_move_towards(tuple(dst))

        key = {
            (-1,0): pygame.K_LEFT,
            ( 1,0): pygame.K_RIGHT,
            ( 0,-1): pygame.K_UP,
            ( 0, 1): pygame.K_DOWN
        }.get(step, pygame.K_SPACE)

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, {'key': key}))
        pygame.event.post(pygame.event.Event(pygame.KEYUP,   {'key': key}))

        time.sleep(0.12)
        return not bool(self.on_hand)

    def __has_reached(self, destination) -> bool:
        return (self.location[0] == destination[0]) and (self.location[1] == destination[1])


def llm_proc(arglist):
    """
    Runs in its own process.  
    • Drains the game→LLM pipe on a background thread  
    • Asks ChatGPT via ask_async() so it never blocks  
    • Parses & execs the reply into a task_queue  
    • Steps through that queue, pressing keys until done
    """
    import queue, threading, sys, time

    from multiprocessing.connection import Listener

    # 1) set up listener + state queue
    listener = Listener(("localhost", 6000))
    state_q: queue.Queue = queue.Queue(maxsize=1)

    def _pipe_reader():
        conn = None
        while True:
            if listener.last_accepted is None:
                conn = listener.accept()
                print(colors.GREEN + "[llm_proc] pipe connected" + colors.ENDC)
            try:
                msg = conn.recv()
            except EOFError:
                break
            # keep only freshest state
            while not state_q.empty():
                state_q.get_nowait()
            state_q.put_nowait(msg)

    threading.Thread(target=_pipe_reader, daemon=True).start()

    def _apply_latest_state():
        while not state_q.empty():
            _, agent_id, loc, act_str, act_loc = state_q.get_nowait()
            if agent_id == 1:
                agent1.set_state(loc, act_str, act_loc)
            else:
                agent2.set_state(loc, act_str, act_loc)

    # 2) instantiate your two agents
    agent1 = LLMAgent(1, arglist)
    agent2 = LLMAgent(2, arglist)

    # 3) pick ChatGPT
    if not arglist.gpt:
        raise RuntimeError("llm_proc only supports --gpt")
    chatbot = ChatGPT(g_openai_config, arglist)

    # 4) ask the user
    sys.stdin = open(0)
    question = input(colors.GREEN + "Enter a task: " + colors.ENDC)
    print(colors.YELLOW + "ChatGPT: thinking (async)…" + colors.ENDC)

    # 5) fire-and-forget LLM call
    reply_q = chatbot.ask_async(question)
    waiting = True

    task_queue = []
    max_steps = 100
    step_i = 0
    task_j = 0
    done = False

    # 6) main loop — never blocks >5ms
    while True:
        time.sleep(0.005)

        if done:
            continue

        if waiting:
            # poll for the LLM reply
            try:
                reply = reply_q.get_nowait()
                # got it! either str or Exception
                if isinstance(reply, Exception):
                    print(colors.RED + "[llm_proc] LLM error:\n", reply, colors.ENDC)
                    done = True
                    continue

                # extract & exec the code
                code = extract_python_code(reply)
                if not code:
                    print(colors.RED + "[llm_proc] ERROR: no code found" + colors.ENDC)
                    done = True
                    continue

                local_env = {"agent1": agent1, "agent2": agent2}
                exec(code, globals(), local_env)
                task_queue = local_env.get("task_queue", [])
                print(colors.GREEN + "[llm_proc] task_queue:", task_queue, colors.ENDC)

                waiting = False
            except queue.Empty:
                _apply_latest_state()
                continue

        # once we have a task queue, step through it
        _apply_latest_state()
        # step_i += 1
        # if step_i >= max_steps:
        #     print(colors.RED + "[llm_proc] GAME OVER" + colors.ENDC)
        #     done = True
        #     continue

        func, arg = task_queue[task_j]
        # switch to chef 1 or 2
        if str(agent1) in str(func):
            agent1.reset_state()
            g_keyboard.press('1'); g_keyboard.release('1')
            while agent1.location is None:
                _apply_latest_state()
        else:
            agent2.reset_state()
            g_keyboard.press('2'); g_keyboard.release('2')
            while agent2.location is None:
                _apply_latest_state()

        if func(arg):
            print(colors.GREEN + f"[llm_proc] completed {func.__name__}({arg})" + colors.ENDC)
            task_j += 1
            if task_j >= len(task_queue):
                print(colors.GREEN + "[llm_proc] ALL TASKS COMPLETE!" + colors.ENDC)
                done = True

class RealAgent:
    """Real Agent object that performs task inference and plans."""

    def __init__(self, arglist, name, id_color, recipes):
        self.arglist = arglist
        self.name = name
        self.color = id_color
        self.recipes = recipes

        # Bayesian Delegation.
        self.reset_subtasks()
        self.new_subtask = None
        self.new_subtask_agent_names = []
        self.incomplete_subtasks = []
        self.signal_reset_delegator = False
        self.is_subtask_complete = lambda w: False
        self.beta = arglist.beta
        self.none_action_prob = 0.5

        self.model_type = agent_settings(arglist, name)
        if self.model_type == "up":
            self.priors = 'uniform'
        else:
            self.priors = 'spatial'

        # Navigation planner.
        self.planner = E2E_BRTDP(
                alpha=arglist.alpha,
                tau=arglist.tau,
                cap=arglist.cap,
                main_cap=arglist.main_cap)

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = Agent(arglist=self.arglist,
                name=self.name,
                id_color=self.color,
                recipes=self.recipes)
        a.subtask = self.subtask
        a.new_subtask = self.new_subtask
        a.subtask_agent_names = self.subtask_agent_names
        a.new_subtask_agent_names = self.new_subtask_agent_names
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def select_action(self, obs):
        """Return best next action for this agent given observations."""
        sim_agent = list(filter(lambda x: x.name == self.name, obs.sim_agents))[0]
        self.location = sim_agent.location
        self.holding = sim_agent.holding
        self.action = sim_agent.action

        if obs.t == 0:
            self.setup_subtasks(env=obs)

        # Select subtask based on Bayesian Delegation.
        self.update_subtasks(env=obs)
        self.new_subtask, self.new_subtask_agent_names = self.delegator.select_subtask(
                agent_name=self.name)
        self.plan(copy.copy(obs))
        return self.action

    def get_subtasks(self, world):
        """Return different subtask permutations for recipes."""
        self.sw = STRIPSWorld(world, self.recipes)
        # [path for recipe 1, path for recipe 2, ...] where each path is a list of actions.
        subtasks = self.sw.get_subtasks(max_path_length=self.arglist.max_num_subtasks)
        all_subtasks = [subtask for path in subtasks for subtask in path]

        # Uncomment below to view graph for recipe path i
        # i = 0
        # pg = recipe_utils.make_predicate_graph(self.sw.initial, recipe_paths[i])
        # ag = recipe_utils.make_action_graph(self.sw.initial, recipe_paths[i])
        return all_subtasks

    def setup_subtasks(self, env):
        """Initializing subtasks and subtask allocator, Bayesian Delegation."""
        self.incomplete_subtasks = self.get_subtasks(world=env.world)
        self.delegator = BayesianDelegator(
                agent_name=self.name,
                all_agent_names=env.get_agent_names(),
                model_type=self.model_type,
                planner=self.planner,
                none_action_prob=self.none_action_prob)

    def reset_subtasks(self):
        """Reset subtasks---relevant for Bayesian Delegation."""
        self.subtask = None
        self.subtask_agent_names = []
        self.subtask_complete = False

    def refresh_subtasks(self, world):
        """Refresh subtasks---relevant for Bayesian Delegation."""
        # Check whether subtask is complete.
        self.subtask_complete = False
        if self.subtask is None or len(self.subtask_agent_names) == 0:
            print("{} has no subtask".format(color(self.name, self.color)))
            return
        self.subtask_complete = self.is_subtask_complete(world)
        print("{} done with {} according to planner: {}\nplanner has subtask {} with subtask object {}".format(
            color(self.name, self.color),
            self.subtask, self.is_subtask_complete(world),
            self.planner.subtask, self.planner.goal_obj))

        # Refresh for incomplete subtasks.
        if self.subtask_complete:
            if self.subtask in self.incomplete_subtasks:
                self.incomplete_subtasks.remove(self.subtask)
                self.subtask_complete = True
        print('{} incomplete subtasks:'.format(
            color(self.name, self.color)),
            ', '.join(str(t) for t in self.incomplete_subtasks))

    def update_subtasks(self, env):
        """Update incomplete subtasks---relevant for Bayesian Delegation."""
        if ((self.subtask is not None and self.subtask not in self.incomplete_subtasks)
                or (self.delegator.should_reset_priors(obs=copy.copy(env),
                            incomplete_subtasks=self.incomplete_subtasks))):
            self.reset_subtasks()
            self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
        else:
            if self.subtask is None:
                self.delegator.set_priors(
                    obs=copy.copy(env),
                    incomplete_subtasks=self.incomplete_subtasks,
                    priors_type=self.priors)
            else:
                self.delegator.bayes_update(
                        obs_tm1=copy.copy(env.obs_tm1),
                        actions_tm1=env.agent_actions,
                        beta=self.beta)

    def all_done(self):
        """Return whether this agent is all done.
        An agent is done if all Deliver subtasks are completed."""
        if any([isinstance(t, Deliver) for t in self.incomplete_subtasks]):
            return False
        return True

    def get_action_location(self):
        """Return location if agent takes its action---relevant for navigation planner."""
        return tuple(np.asarray(self.location) + np.asarray(self.action))

    def plan(self, env, initializing_priors=False):
        """Plan next action---relevant for navigation planner."""
        print('right before planning, {} had old subtask {}, new subtask {}, subtask complete {}'.format(self.name, self.subtask, self.new_subtask, self.subtask_complete))

        # Check whether this subtask is done.
        if self.new_subtask is not None:
            self.def_subtask_completion(env=env)

        # If subtask is None, then do nothing.
        if (self.new_subtask is None) or (not self.new_subtask_agent_names):
            actions = nav_utils.get_single_actions(env=env, agent=self)
            probs = []
            for a in actions:
                if a == (0, 0):
                    probs.append(self.none_action_prob)
                else:
                    probs.append((1.0-self.none_action_prob)/(len(actions)-1))
            self.action = actions[np.random.choice(len(actions), p=probs)]
        # Otherwise, plan accordingly.
        else:
            if self.model_type == 'greedy' or initializing_priors:
                other_agent_planners = {}
            else:
                # Determine other agent planners for level 1 planning.
                # Other agent planners are based on your planner---agents never
                # share planners.
                backup_subtask = self.new_subtask if self.new_subtask is not None else self.subtask
                other_agent_planners = self.delegator.get_other_agent_planners(
                        obs=copy.copy(env), backup_subtask=backup_subtask)

            print("[ {} Planning ] Task: {}, Task Agents: {}".format(
                self.name, self.new_subtask, self.new_subtask_agent_names))

            action = self.planner.get_next_action(
                    env=env, subtask=self.new_subtask,
                    subtask_agent_names=self.new_subtask_agent_names,
                    other_agent_planners=other_agent_planners)

            # If joint subtask, pick your part of the simulated joint plan.
            if self.name not in self.new_subtask_agent_names and self.planner.is_joint:
                self.action = action[0]
            else:
                self.action = action[self.new_subtask_agent_names.index(self.name)] if self.planner.is_joint else action

        # Update subtask.
        self.subtask = self.new_subtask
        self.subtask_agent_names = self.new_subtask_agent_names
        self.new_subtask = None
        self.new_subtask_agent_names = []

        print('{} proposed action: {}\n'.format(self.name, self.action))

    def def_subtask_completion(self, env):
        # Determine desired objects.
        self.start_obj, self.goal_obj = nav_utils.get_subtask_obj(subtask=self.new_subtask)
        self.subtask_action_object = nav_utils.get_subtask_action_obj(subtask=self.new_subtask)

        # Define termination conditions for agent subtask.
        # For Deliver subtask, desired object should be at a Deliver location.
        if isinstance(self.new_subtask, Deliver):
            self.cur_obj_count = len(list(
                filter(lambda o: o in set(env.world.get_all_object_locs(self.subtask_action_object)),
                env.world.get_object_locs(obj=self.goal_obj, is_held=False))))
            self.has_more_obj = lambda x: int(x) > self.cur_obj_count
            self.is_subtask_complete = lambda w: self.has_more_obj(
                    len(list(filter(lambda o: o in
                set(env.world.get_all_object_locs(obj=self.subtask_action_object)),
                w.get_object_locs(obj=self.goal_obj, is_held=False)))))
        # Otherwise, for other subtasks, check based on # of objects.
        else:
            # Current count of desired objects.
            self.cur_obj_count = len(env.world.get_all_object_locs(obj=self.goal_obj))
            # Goal state is reached when the number of desired objects has increased.
            self.is_subtask_complete = lambda w: len(w.get_all_object_locs(obj=self.goal_obj)) > self.cur_obj_count


class SimAgent:
    """Simulation agent used in the environment object."""

    def __init__(self, name, id_color, location):
        self.name = name
        self.color = id_color
        self.location = location
        self.holding = None
        self.action = (0, 0)
        self.has_delivered = False

    def __str__(self):
        return color(self.name[-1], self.color)

    def __copy__(self):
        a = SimAgent(name=self.name, id_color=self.color,
                location=self.location)
        a.__dict__ = self.__dict__.copy()
        if self.holding is not None:
            a.holding = copy.copy(self.holding)
        return a

    def get_repr(self):
        return AgentRepr(name=self.name, location=self.location, holding=self.get_holding())

    def get_holding(self):
        if self.holding is None:
            return 'None'
        return self.holding.full_name

    def print_status(self):
        print("{} currently at {}, action {}, holding {}".format(
                color(self.name, self.color),
                self.location,
                self.action,
                self.get_holding()))

    def acquire(self, obj):
        if self.holding is None:
            self.holding = obj
            self.holding.is_held = True
            self.holding.location = self.location
        else:
            self.holding.merge(obj) # Obj(1) + Obj(2) => Obj(1+2)

    def release(self):
        self.holding.is_held = False
        self.holding = None

    def move_to(self, new_location):
        self.location = new_location
        if self.holding is not None:
            self.holding.location = new_location