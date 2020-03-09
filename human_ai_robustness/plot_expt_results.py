
from human_aware_rl.ppo.ppo_pop import plot_ppo_tom

# SEEDS = [3264, 4859, 9225, 2732, 9845]
SEEDS = [2732]

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/hp_tune_sce1'
# DIR = '/home/pmzpk/Dropbox/Pycharm_paulk444/human_ai_robustness/human_ai_robustness/data/actual_experiments'

for i in range(4, 8):

    # run_name = 'sce_{}'.format(i)
    run_name = 'sce_20toms_{}l'.format(i)
    # run_name = 'tom_pop_expt/cring_20_toms_p'

    plot_ppo_tom(DIR, run_name, SEEDS)