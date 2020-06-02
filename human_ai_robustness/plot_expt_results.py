
from human_aware_rl.ppo.ppo_pop import plot_ppo_tom
import matplotlib.pyplot as plt

SEEDS = [2732, 3264, 4859, 9225, 9845]
# SEEDS = [2732, 4859, 9845]
# SEEDS = [9845]

map = 'cpot'

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/neu_expt_cobj_cpot0'.format(map)
# DIR = '/home/pmzpk/Dropbox/Pycharm_paulk444/human_ai_robustness/human_ai_robustness/data/'

# runs = ['{}'.format(i) for i in range(11)]
# map = 'cc'
runs = ['1tom', '20tom', '1bc', '20bc']
# runs = ['2b', '2c', '3b', '3c', '1b', '7b', '7c', '12', '13', '5c', '5b']

for i in range(len(runs)):

    run_name = '{}_{}'.format(map, runs[i])
    # run_name = 'tom_pop_expt/cring_20_toms_p'

    # SEEDS = SEEDS if run_name != 'cring_20mixed' else [2732, 3264, 4859, 9225]

    plot_ppo_tom(DIR, run_name, SEEDS, plot_val=False, plot_dense=False)

plt.show()
