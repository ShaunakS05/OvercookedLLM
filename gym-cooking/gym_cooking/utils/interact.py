from utils.core import *
import numpy as np
from utils.core import cook_object, Skillet

def interact(agent, world):
    """Carries out interaction for this agent taking this action in this world.

    The action that needs to be executed is stored in `agent.action`.
    """

    # agent does nothing (i.e. no arrow key)
    if agent.action == (0, 0):
        return

    # Save direction if this was a movement key
    if agent.action in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        agent.last_direction = agent.action
        # Move agent to floor tile, no interaction
        target = tuple(np.asarray(agent.location) + np.asarray(agent.action))
        x, y = world.inbounds(target)
        gs = world.get_gridsquare_at((x, y))
        if isinstance(gs, Floor):
            agent.move_to(gs.location)
        return

    # Handle interaction only if explicitly triggered
    if agent.action != "interact":
        return

    # Make sure agent has a direction to interact toward
    if not hasattr(agent, 'last_direction'):
        return  # can't interact without facing direction

    dx, dy = agent.last_direction
    target = tuple(np.asarray(agent.location) + np.asarray((dx, dy)))
    x, y = world.inbounds(target)
    gs = world.get_gridsquare_at((x, y))
    if gs is None:
        return

    # if holding something
    if agent.holding is not None:
        # if delivery in front --> deliver
        if isinstance(gs, Delivery):
            obj = agent.holding
            if obj.is_deliverable():
                gs.acquire(obj)
                agent.release()
                print('\nDelivered {}!'.format(obj.full_name))

        # if occupied gridsquare in front --> try merging
        elif world.is_occupied(gs.location):
            other = world.get_object_at(gs.location, None, find_held_objects=False)
            if mergeable(agent.holding, other):
                # take the counter object off the grid
                gs_obj = gs.release()     
                world.remove(gs_obj)
                # remove the agent’s held object from the world registry
                held_obj = agent.holding
                world.remove(held_obj)
                # merge them into a single Object
                gs_obj.merge(held_obj)
                # now that gs_obj has both contents, agent picks it up
                agent.acquire(gs_obj)
                world.insert(gs_obj)
                # if you’re in “play” mode you also want it to stay on the counter, so re‐drop:
                if world.arglist.play:
                    gs.acquire(agent.holding)
                    agent.release()

        # if holding something, empty gridsquare in front --> chop or drop
        elif not world.is_occupied(gs.location):
            obj = agent.holding
            if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
                obj.chop()
            else:
                if isinstance(gs, Skillet):
                    obj.cook_timer = world.env_time + 450
                    print(f"[DEBUG] Timer set at {obj.cook_timer}, now={world.env_time}")
                gs.acquire(obj)
                agent.release()
                assert not world.get_object_at(gs.location, obj, find_held_objects=False).is_held, "Verifying put down works"

    # if not holding anything
    elif agent.holding is None:
        if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
            obj = world.get_object_at(gs.location, None, find_held_objects=False)
            if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
                obj.chop()
            else:
                gs.release()
                agent.acquire(obj)
        elif not world.is_occupied(gs.location):
            pass


    # # if holding something
    # elif agent.holding is not None:
    #     # if delivery in front --> deliver
    #     if isinstance(gs, Delivery):
    #         obj = agent.holding
    #         if obj.is_deliverable():
    #             gs.acquire(obj)
    #             agent.release()
    #             print('\nDelivered {}!'.format(obj.full_name))

    #     # if occupied gridsquare in front --> try merging
    #     elif world.is_occupied(gs.location):
    #         # Get object on gridsquare/counter
    #         obj = world.get_object_at(gs.location, None, find_held_objects = False)

    #         if mergeable(agent.holding, obj):
    #             world.remove(obj)
    #             o = gs.release() # agent is holding object
    #             world.remove(agent.holding)
    #             agent.acquire(obj)
    #             world.insert(agent.holding)
    #             # if playable version, merge onto counter first
    #             if world.arglist.play:
    #                 gs.acquire(agent.holding)
    #                 agent.release()


    #     # if holding something, empty gridsquare in front --> chop or drop
    #     elif not world.is_occupied(gs.location):
    #         obj = agent.holding
    #         if isinstance(gs, Cutboard) and obj.needs_chopped() and not world.arglist.play:
    #             # normally chop, but if in playable game mode then put down first
    #             obj.chop()
    #         else:
    #             gs.acquire(obj) # obj is put onto gridsquare
    #             agent.release()
    #             assert world.get_object_at(gs.location, obj, find_held_objects =\
    #                 False).is_held == False, "Verifying put down works"

    # if not holding anything
    # elif agent.holding is None:
    #     # not empty in front --> pick up
    #     if world.is_occupied(gs.location) and not isinstance(gs, Delivery):
    #         obj = world.get_object_at(gs.location, None, find_held_objects = False)
    #         # if in playable game mode, then chop raw items on cutting board
    #         if isinstance(gs, Cutboard) and obj.needs_chopped() and world.arglist.play:
    #             obj.chop()
    #         else:
    #             gs.release()
    #             agent.acquire(obj)

    #     # if empty in front --> interact
    #     elif not world.is_occupied(gs.location):
    #         pass
