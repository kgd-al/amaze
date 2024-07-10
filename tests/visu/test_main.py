import sys

import pytest
from PyQt5.QtGui import QImage

from _common import EventObserver
from amaze import amaze_main as main, Maze
from amaze.misc.utils import qt_offscreen, qt_application, has_qt_application, NoQtApplicationException, \
    is_qt_offscreen, QT_PLATFORM_OFFSCREEN_PLUGIN, QtOffscreen

BUILTIN_CONTROLLER = "cheater"
CONTROLLER_FILE = "examples/agents/unicursive_tabular.zip"


def _main(*args, maze=True, controller=True, file_controller=False):
    _args = ["-v"]
    if maze:
        _args.extend(["--maze", "M10_10x10_C1"])
    if controller:
        _args.append("--controller")
        if file_controller:
            _args.append(CONTROLLER_FILE)
        else:
            _args.append(BUILTIN_CONTROLLER)
    return main(_args + list(*args))


def test_app():
    pytest.raises(NoQtApplicationException, has_qt_application)
    pytest.raises(RuntimeError, qt_application, False)
    qt_offscreen(offscreen=False)
    assert not is_qt_offscreen()
    app = qt_application(allow_create=True, start_offscreen=False)
    assert has_qt_application()
    assert app == qt_application(allow_create=False, start_offscreen=False)
    assert not is_qt_offscreen()
    qt_offscreen(offscreen=True)
    assert is_qt_offscreen()
    qt_offscreen(offscreen=False)
    assert not is_qt_offscreen()
    del app

    pytest.raises(NoQtApplicationException, has_qt_application)
    app = qt_application(allow_create=True, start_offscreen=True)
    assert is_qt_offscreen()
    assert app.platformName() == QT_PLATFORM_OFFSCREEN_PLUGIN
    del app

    with QtOffscreen() as app:
        assert is_qt_offscreen()
    assert not is_qt_offscreen()
    del app


def test_main_help():  # argparse help exits instead of returning
    with pytest.raises(SystemExit) as e:
        main("-h")
    assert e.value.code == 0


def test_main_no_args():
    with QtOffscreen() as app:
        eo = EventObserver(app)
        assert main([]) == 0


def test_main_no_sys_args():
    with QtOffscreen() as app:
        eo = EventObserver(app)
        argv = sys.argv
        sys.argv = ["Highjacked sys.argv"]
        assert main() == 0
        sys.argv = argv


def test_main_verbose(capfd):
    with QtOffscreen() as app:
        eo = EventObserver(app)
        assert main(["-v"]) == 0
        assert "Executing with" in capfd.readouterr().out


def test_main_eval(tmp_path):
    assert 0 == _main(["--evaluate", tmp_path], file_controller=False)
    assert 0 == _main(["--evaluate", tmp_path], file_controller=True)
    assert len(list(tmp_path.glob("*"))) == 0


def _test_render_file(path, count=1):
    assert len(list(path.parent.glob("*"))) == count
    assert path.exists()
    assert not QImage(str(path)).isNull()


def test_main_render(tmp_path):
    maze = tmp_path.joinpath("maze.png")
    assert 0 == _main(["--render", tmp_path])  # Fails
    assert 0 == _main(["--render", maze])      # Succeeds


def _test_trajectory_file(path, count=1):
    assert len(list(path.parent.glob("*"))) == count
    assert path.exists()
    assert not QImage(str(path)).isNull()


def test_main_trajectory(tmp_path):
    traj = tmp_path.joinpath("trajectory.png")
    assert 0 == _main(["--trajectory", traj])
    _test_trajectory_file(traj)


def _test_eval_inputs_files(tmp_path, count=2):
    assert len(list(tmp_path.glob("*"))) == count
    i_img = tmp_path.joinpath("inputs_recognition.png")
    assert i_img.exists()
    assert not QImage(str(i_img)).isNull()
    i_csv = tmp_path.joinpath("inputs_recognition.csv")
    assert i_csv.exists()
    with open(i_csv, "rt") as f:
        assert len(f.readlines())


def test_main_eval_inputs(tmp_path):
    assert 0 == _main(["--evaluate-inputs", tmp_path], file_controller=True)
    _test_eval_inputs_files(tmp_path)


def _test_movie_files(path, count=2):
    assert len(list(path.parent.glob("*"))) == count
    assert path.exists()
    i_folder = path.with_suffix("")
    assert i_folder.exists()


def test_main_movie(tmp_path):
    movie = tmp_path.joinpath("movie.gif")
    assert 0 == _main(["--movie", movie])
    _test_movie_files(movie)


def _test_all(args, flags, tmp_path):
    maze_file = "maze.png"
    traj_file = "trajectory.png"
    files = 4
    assert 0 == _main([
            "--evaluate-inputs", tmp_path,
            "--evaluate", tmp_path,
            "--render", maze_file,
            "--trajectory", traj_file,
            "--dark"
        ] + args,
        file_controller=True, **flags)
    _test_render_file(tmp_path.joinpath(maze_file), files)
    _test_trajectory_file(tmp_path.joinpath(traj_file), files)
    _test_eval_inputs_files(tmp_path, files)


@pytest.mark.parametrize("args, flags", [
    pytest.param([], dict(), id="base"),
    pytest.param(["--maze", "M9_9x9_t1_T1"], dict(maze=False), id="maze"),
    pytest.param(["--maze-seed", "9", "--maze-width", "10", "--maze-height", "11"],
                 dict(maze=False), id="maze_argv"),
    pytest.param(["--robot", "H"], dict(), id="robot"),
    pytest.param(["--controller", "random"], dict(controller=False), id="controller"),
    pytest.param(["--cell-width", "10"], dict(), id="cell-width"),
])
def test_main_eval_all(args, flags, tmp_path):
    _test_all(args, flags, tmp_path)


def test_main_eval_mazes(mbd_kwargs, tmp_path):
    m_str = Maze.BuildData(**mbd_kwargs).to_string()
    _test_all(["--maze", m_str], dict(maze=False), tmp_path)


def test_main_error(tmp_path):
    with pytest.raises(ValueError, match="Cannot evaluate without a controller"):
        main(["--evaluate", tmp_path])

    with pytest.raises(ValueError, match="Cannot plot trajectory without a controller"):
        main(["--trajectory", tmp_path])

    with pytest.raises(ValueError, match="No controller archive found"):
        main(["--controller", tmp_path.joinpath("foo.zip")])

    with pytest.raises(ValueError, match="Cannot evaluate inputs without a controller"):
        main(["--evaluate-inputs", tmp_path.joinpath("foo")])

    # assert False
