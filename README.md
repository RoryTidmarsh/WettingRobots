# WettingRobots
 
"basic viscek files" folder: the original code given by F.Turci, a jupyter notebook of me understanding this code and our break down of the code using 'for' loops.

"wall implementation" folder looks at implementing a wall into the viscek model. It contains:
- "Viscek_loops_wall optimising": **MAIN WALL IMPLEMENTATION**. The wall implemented using numba to speed up
 - "Vicesek_looped_walls.py": the initial implementation of the wall using for loops.
 - "shapes.py" and "Viscek_wall_importingShapes.py": me trying out different shapes by containing relevant functions in classes in "shapes.py" - doesn't work with numba.
 - "wall implementation.ipynb": jupyternotebook of me figuring this all out.
