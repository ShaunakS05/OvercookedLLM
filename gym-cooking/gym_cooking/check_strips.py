import recipe_planner.recipe as rec
import recipe_planner.stripsworld as sw

# --- minimal empty world stub ---------------------------------
class W: 
    objects = {}
    def get_object_list(self):        # STRIPS never calls it here
        return []

# --- ask STRIPSWorld for a plan --------------------------------
world = W()
simple_burger = rec.SimpleBurger()
sworld = sw.STRIPSWorld(world, [simple_burger])
subtasks = sworld.get_subtasks(max_path_length=12)
print("\nSubtasks returned:\n", subtasks)
