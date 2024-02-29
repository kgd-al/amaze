from amaze import *

FOLDER = "tmp/demos/visualization"
WIDTH = 200  # Of generated images

maze_str = "20x20_U"
simulation = Simulation(
    Maze.from_string(maze_str),
    Robot.BuildData.from_string("DD"),
    save_trajectory=True
)
maze_str = simulation.maze.to_string()

app = application()
widget = MazeWidget(simulation,
                    config=dict(
                        robot=False, solution=True,
                        dark=True, colorblind=True
                    ), width=200)

maze_img = f"{FOLDER}/{maze_str}.png"
if widget.draw_to(maze_img):
    print(f"Saved {simulation.maze.to_string()} to {maze_img}")
del widget

agent_path = "examples/agents/unicursive_tabular.zip"
controller = load(agent_path)
simulation.run(controller)

trajectory_img = f"{FOLDER}/{agent_path.split('/')[-1].split('.')[0]}_{maze_str}.png"
MazeWidget.plot_trajectory(
    simulation=simulation, size=WIDTH, path=trajectory_img,
)
print(f"Plotted {agent_path}"
      f" in {simulation.maze.to_string()}"
      f" to {trajectory_img}")

main(["--maze", maze_str, "--controller", "random", "--auto-quit"])
