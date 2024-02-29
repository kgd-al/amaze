from amaze import *

# Global variables
FOLDER = "tmp/demos/visualization"
WIDTH = 512  # Of generated images

# Define a seed-less maze and get the resolved name
maze_str = "10x10_U"
maze = Maze.from_string(maze_str)
maze_str = maze.to_string()

# Draw the maze
app = application()
maze_img = f"{FOLDER}/{maze_str}.png"
if MazeWidget.draw_to(
        maze=Maze.from_string(maze_str),
        path=maze_img, size=WIDTH,
        colorblind=True, robot=False, solution=True, dark=True):
    print(f"Saved {maze_str} to {maze_img}")

# Have an agent move around in the maze ...
agent_path = "examples/agents/unicursive_tabular.zip"
controller = load(agent_path)
simulation = Simulation(
    maze,
    Robot.BuildData.from_string("DD"),
    save_trajectory=True
)
simulation.run(controller)

# ... and print its trajectory
agent_name = agent_path.split('/')[-1].split('.')[0]
trajectory_img = f"{FOLDER}/{agent_name}_{maze_str}.png"
MazeWidget.plot_trajectory(
    simulation=simulation, size=WIDTH, path=trajectory_img,
)
print(f"Plotted {agent_path}" 
      f" in {simulation.maze.to_string()}"
      f" to {trajectory_img}")

# Invoke the main from python (with arguments)
amaze_main(["--maze", maze_str, "--controller", "random", "--auto-quit"])
