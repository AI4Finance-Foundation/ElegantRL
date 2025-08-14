from torch.utils.data import DataLoader

from rlsolver.methods.eco_s2v.rl4co.envs import (
    CVRPEnv,
    CVRPTWEnv,
    DPPEnv,
    FLPEnv,
    MCPEnv,
    MDPPEnv,
    MTSPEnv,
    OPEnv,
    PCTSPEnv,
    PDPEnv,
    PDPRuinRepairEnv,
    SDVRPEnv,
    SMTWTPEnv,
    SPCTSPEnv,
    TSPEnv,
)


def get_env(name, size):
    match name:
        case "tsp":
            env = TSPEnv(generator_params=dict(num_loc=size))
        case "cvrp":
            env = CVRPEnv(generator_params=dict(num_loc=size))
        case "cvrptw":
            env = CVRPTWEnv(generator_params=dict(num_loc=size))
        case "sdvrp":
            env = SDVRPEnv(generator_params=dict(num_loc=size))
        case "pdp":
            env = PDPEnv(generator_params=dict(num_loc=size))
        case "op":
            env = OPEnv(generator_params=dict(num_loc=size))
        case "mtsp":
            env = MTSPEnv(generator_params=dict(num_loc=size))
        case "pctsp":
            env = PCTSPEnv(generator_params=dict(num_loc=size))
        case "spctsp":
            env = SPCTSPEnv(generator_params=dict(num_loc=size))
        case "dpp":
            env = DPPEnv()
        case "mdpp":
            env = MDPPEnv()
        case "smtwtp":
            env = SMTWTPEnv()
        case "pdp_ruin_repair":
            env = PDPRuinRepairEnv()
        case "mcp":
            env = MCPEnv()
        case "flp":
            env = FLPEnv()
        case _:
            raise ValueError(f"Unknown env_name: {name}")

    return env.transform()


def generate_env_data(env, size, batch_size):
    env = get_env(env, size)
    dataset = env.dataset([batch_size])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=dataset.collate_fn,
    )

    return env, next(iter(dataloader))
