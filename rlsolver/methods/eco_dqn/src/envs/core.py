from rlsolver.methods.eco_dqn.src.envs.spinsystem import SpinSystemFactory

def make(id, *args, **kwargs):

    if id == "SpinSystem":
        env = SpinSystemFactory.get(*args, **kwargs)

    else:
        raise NotImplementedError()

    return env