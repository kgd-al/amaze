import itertools

from common import Progress, STEPS

from procgen.gym_registration import make_env as make_procgen_env, ENV_NAMES as PROCGEN_ENV_NAMES

def _evaluate(env_name, **kwargs):
    env = make_procgen_env(env_name=env_name, **kwargs)
    env.reset()

    for _ in range(STEPS):
        action = env.action_space.sample()
        env.step(action)

    env.close()
    return True


def process(df):
    procgen_args = []
    for name in PROCGEN_ENV_NAMES:
        difficulties = ["easy", "hard"]
        if name in ["chaser", "dodgeball", "leaper", "starpilot"]:
            difficulties.append("extreme")
        if name in ["caveflyer", "dodgeball", "heist", "jumper", "maze", "miner"]:
            difficulties.append("memory")
        if name in ["coinrun", "caveflyer", "leaper", "jumper", "maze", "heist", "climber", "ninja"]:
            difficulties.append("exploration")
        procgen_args.extend([
            dict(name=name, distribution_mode=d, use_backgrounds=t, restrict_themes=t, use_monochrome_assets=t)
            for d, t in itertools.product(difficulties, [True, False])])
    #pprint.pprint(procgen_args)
    #print(PROCGEN_ENV_NAMES, len(PROCGEN_ENV_NAMES))

    with Progress(df, "misc") as progress:
        progress.add_task(len(procgen_args))

        for args in procgen_args:
            name = args.pop("name")
            difficulty = args["distribution_mode"]
            candidate = (
                f"Procgen/" + name + "-" + difficulty
                + "".join(str(int(b)) for b in args.values() if isinstance(b, bool))
            )

            progress.evaluate("ProcGen", candidate, _evaluate, env_name=name, **args)

        progress.close()

    return progress.errors
