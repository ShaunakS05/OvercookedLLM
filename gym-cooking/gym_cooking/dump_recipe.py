import importlib, recipe_planner.recipe as rec
import textwrap

# reload to be sure we run the latest edits
importlib.reload(rec)

r = rec.SimpleBurger()

print("\n--- ACTIONS IN SimpleBurger --------------------------------")
for a in sorted(r.actions, key=lambda x: x.name):
    print(f"\n{a}")                        # short name  e.g. Cook(Burger)
    print(textwrap.indent(a.specs, "  ")) # full spec (pre / post)
