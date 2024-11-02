from rlsolver.methods.eco_s2v.src.envs.spinsystem import SpinSystemFactory


def make(id2, *args, **kwargs):
    if id2 == "SpinSystem":
        env = SpinSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env
