
from human_aware_rl.ppo.ppo_pop import plot_ppo_tom

# SEEDS = [3264, 4859, 9225, 2732, 9845]
SEEDS = [2732]

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/all_toms_cc0'
# DIR = '/home/pmzpk/Dropbox/Pycharm_paulk444/human_ai_robustness/human_ai_robustness/data/'

runs = ['tom_0_quick']
map = 'cc'

for i in range(len(runs)):

    run_name = '{}_{}'.format(map, runs[i])
    # run_name = 'tom_pop_expt/cring_20_toms_p'

    plot_ppo_tom(DIR, run_name, SEEDS, plot_val=False)
