
import numpy as np
from human_ai_robustness.bc_experiments_from_human_model_theory import train_bc_models
from argparse import ArgumentParser

# Training params:
params_croom = {"layout_name": "cramped_room", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
params_aa = {"layout_name": "asymmetric_advantages", "num_epochs": 100, "lr": 1e-3, "adam_eps":1e-8}
params_cring = {"layout_name": "coordination_ring", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
# params_random0 = {"layout_name": "forced_coordination", "num_epochs": 90, "lr": 1e-3, "adam_eps":1e-8}
params_cc = {"layout_name": "counter_circuit", "num_epochs": 110, "lr": 1e-3, "adam_eps":1e-8}
all_params = [params_croom, params_aa, params_cring, params_cc]

if __name__ == "__main__":
    """
    """

    parser = ArgumentParser()
    # parser.add_argument("-l", "--fixed_mdp", dest="layout",
    #                     help="name of the layout to be played as found in data/layouts",
    #                     required=True)
    parser.add_argument("-ns", "--num_seeds", dest="num_seeds",
                        help="Number of seeds -- i.e. number of bc models to train", required=True, type=int)
    # parser.add_argument("-tm", "--time_limit", default=30, type=float)
    # parser.add_argument("-pd", "--ppo_dir", required=False, type=str)

    args = parser.parse_args()
    num_seeds = args.num_seeds

    seeds = list(np.random.randint(0, 10000, size=num_seeds))
    train_bc_models(all_params, seeds, data_subset="train", prefix="", evaluation=True)
