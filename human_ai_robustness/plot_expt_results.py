
from human_aware_rl.ppo.ppo_tom import plot_ppo_tom

# SEEDS = [3264, 4859, 9225]
SEEDS = [2732, 9845]

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/hp_tune_cring4'
# DIR = /home/pmzpk/Dropbox/Pycharm_paulk444/human_ai_robustness/human_ai_robustness/data/actual_experiments

for i in range(8):

    run_name = 'cring_' + str(i)
    # run_name = 'aa_2'

    plot_ppo_tom(DIR, run_name, SEEDS)