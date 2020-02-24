
from human_aware_rl.ppo.ppo_pop import plot_ppo_tom

SEEDS = [3264, 4859, 9225, 2732, 9845]

# DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/pop_expt_cring'
DIR = '/home/pmzpk/Dropbox/Pycharm_paulk444/human_ai_robustness/human_ai_robustness/data/actual_experiments'

for i in range(1):

    # run_name = 'cring_' + str(i)
    run_name = 'tom_pop_expt/cring_20_toms_p'

    plot_ppo_tom(DIR, run_name, SEEDS)