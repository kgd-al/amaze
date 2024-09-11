import pprint
import gymnasium as gym
import ale_py

from common import line, Progress, STEPS


registration = gym.envs.registration

def _short_name(item):
    return item.name.replace("Jax-", "").split("-")[0].split("_")[0].replace("Deterministic", "").replace("NoFrameskip", "")


def _evaluate(env_name):
    env = gym.make(env_name)
    observation, info = env.reset()

    for _ in range(STEPS):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


def process(df):
    # ===========================================================================================
    # == Candidates collection
    # ===========================================================================================

    #gym.pprint_registry()
    registry = registration.registry
    ordered_registry = {}
    all_kwargs, kwargs_map = {}, {}
    short_names, names = set(), {}
    for name, item in registry.items():
        key = _short_name(item)

        if item.namespace in ["dm_control", "phys2d", "tabular"]:
            continue

        if "jax" in key.lower():
            continue

        if key in ["GymV21Environment", "GymV26Environment", "Combat"]:
            continue

        if item.namespace == "ALE" and key in ["Combat", "Joust", "MazeCraze", "Warlords"]:
            continue

        names[item.name] = key
        short_names.add(key)

        #print(key)
        #pprint.pprint(item)
        version = item.version

        family = item.namespace
        if family is None:
            for env_name, envs in [
                ("Box2D", ["BipedalWalker", "CarRacing", "LunarLander"]),
                ("Toy Text", ["Blackjack", "CliffWalking", "FrozenLake", "Taxi"]),
                ("Mujoco", ["Ant", "HalfCheetah", "Hopper", "Humanoid", "HumanoidStandup",
                            "InvertedDoublePendulum", "InvertedPendulum", "Pusher", "Reacher",
                            "Swimmer", "Walker2d"]),
                ("Classic Control", ["Acrobot", "CartPole", "MountainCar", "Pendulum"]),
            ]:
                if any(n in key for n in envs):
                    family = env_name
                    #print(name, family, item.entry_point)
                    break

            if family is None and isinstance(item.entry_point, str):
                if item.entry_point.split(":")[1] == "AtariEnv":
                    continue
                else:
                    family = item.entry_point.split(".")[-1].split(":")[0]

        if family in ["openai_gym_compatibility", "atari_env"]:
            continue

        #print(f"{key:>30s} v{version} {'('+item.name+')':^40s} from {family}")
        assert family is not None, f"Family not found: {pprint.pformat(item)}"
        assert family[0].upper() == family[0], f"{family=}\n{pprint.pformat(item)}"

        if family not in ordered_registry:
            ordered_registry[family] = {}

        if key not in ordered_registry[family]:
            ordered_registry[family][key] = [0, set(), set(), []]

        kwargs = tuple(sorted(item.kwargs.keys()))

        if kwargs not in kwargs_map:
            kwargs_map[kwargs] = [len(kwargs_map), 0]
        kwargs_map[kwargs][1] += 1

        entry = ordered_registry[family][key]
        entry[0] += 1
        entry[1].add(kwargs_map[kwargs][0])
        entry[2].add(item.namespace)
        entry[3].append((version, name))

    #pprint.pprint(ordered_registry)

    candidates = {}
    for family, entries in ordered_registry.items():
        f_candidates = []
        for short_name, data in sorted(entries.items()):
            #pprint.pprint(data)
            n, kwargs, namespaces, details = data
            assert len(namespaces) == 1, f"{namespaces=}"
            namespace = next(iter(namespaces))
            if n > 1:
                v = registration.find_highest_version(namespace, short_name)
                candidate = f"{short_name}-v{v}"
                if namespace:
                    candidate = f"{namespace}/{candidate}"
            else:
                candidate = details[0][1]
            f_candidates.append(candidate)

            for kw in kwargs:
                if family not in all_kwargs:
                    all_kwargs[family] = {}

                _nkw = all_kwargs[family]
                if kw not in _nkw:
                    _nkw[kw] = 0

                _nkw[kw] += 1

        candidates[family] = f_candidates

    #pprint.pprint({k: v[0:3] for k,v in ordered_registry.items()})
    #pprint.pprint(kwargs_map)
    print(f"{len(registry)} registered environments")
    print("Candidate environments:")
    for family, f_candidates in candidates.items():
        print(f"{len(f_candidates):3d} {family}")
    #line('-')
    #print("keyword arguments:")
    #pprint.pprint(all_kwargs)
    #pprint.pprint(kwargs_map)


    # ===========================================================================================
    # == Eval
    # ===========================================================================================
    line("-")

    with Progress(df, "Gymnasium") as progress:
        progress.add_task(sum(len(fc) for fc in candidates.values()))

        for family, f_candidates in candidates.items():
            for candidate in f_candidates:
                progress.evaluate(family, candidate, _evaluate, env_name=candidate)

        progress.close()

    return progress.errors
