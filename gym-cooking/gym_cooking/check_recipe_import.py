import traceback
try:
    import recipe_planner.recipe as r
    print("✔ recipe_planner.recipe imported fully")
    print("• names with 'Burger':", [n for n in dir(r) if 'Burger' in n])
except Exception:
    print("✗ Import aborted — traceback below:\n")
    traceback.print_exc()