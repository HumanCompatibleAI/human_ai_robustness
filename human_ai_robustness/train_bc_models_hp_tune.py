import time

import numpy as np
from human_ai_robustness.bc_experiments_from_human_model_theory import train_bc_models
from argparse import ArgumentParser

# Training params:
params_croom = {"layout_name": "cramped_room", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
params_aa = {"layout_name": "asymmetric_advantages", "num_epochs": 100, "lr": 1e-3, "adam_eps":1e-8}
params_cring = {"layout_name": "coordination_ring", "num_epochs": 120, "lr": 1e-3, "adam_eps":1e-8}
# params_random0 = {"layout_name": "forced_coordination", "num_epochs": 90, "lr": 1e-3, "adam_eps":1e-8}
params_cc = {"layout_name": "counter_circuit", "num_epochs": 110, "lr": 1e-3, "adam_eps":1e-8}
# all_params = [params_croom, params_aa, params_cring, params_cc]

all_params_test = [{"layout_name": "cramped_room", "num_epochs": 2, "lr": 1e-3, "adam_eps":1e-8}]

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
    parser.add_argument("-t", "--test", required=False, default=False)
    parser.add_argument("-l", "--layouts", required=False, default=False)

    args = parser.parse_args()
    num_seeds, test, layouts = args.num_seeds, args.test, args.layouts

    if test:
        all_params = all_params_test
    else:
        if layouts == 'cring':
            # Only tune HPs for 1 layout:
            all_params = [params_cring]
    
    # HPs to try
    HPs = [ {"lr": 1e-3, "adam_eps": 1e-8},
            {"lr": 1e-2, "adam_eps": 1e-8}]#,
            # {"lr": 1e-4, "adam_eps": 1e-8},
            # {"lr": 1e-3, "adam_eps": 1e-7},
            # {"lr": 1e-3, "adam_eps": 1e-9}]

    np.random.seed(0)
    seeds = list(np.random.randint(0, 10000, size=num_seeds))
    print('Seeds of the trained models:\n{}'.format(seeds))

    training_losses = []
    unstuck_bcbc_rews = []

    for HP in HPs:
        all_params[0]["lr"] = HP["lr"]
        all_params[0]["adam_eps"] = HP["adam_eps"]
        start_time = time.time()
        infos = train_bc_models(all_params, seeds, data_subset="train", prefix="", evaluation=True, print_infos=False, return_infos=True)
        training_losses.append([infos[i]['train_losses'] for i in range(len(seeds))])
        unstuck_bcbc_rews.append([infos[i]['avg_unstuckBCBC_rewards'] for i in range(len(seeds))])
        print('\nCompleted BC with these HPs in {} min(s)\n'.format(round((time.time() - start_time)/60)))

    # Print results:
    for i, HP in enumerate(HPs):
        print('\nHPs: {}; Training losses: {}; Unstuck BCBC rewards: {}'.format(HP, training_losses[i], ))
