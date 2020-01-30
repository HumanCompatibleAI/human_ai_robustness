
from human_aware_rl.ppo.ppo_tom import plot_ppo_tom

SEEDS = [2732, 9845]

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/hp_tune_except_croom'

for i in [0, 1, 2, 4, 5]:

    run_name = 'aa_' + str(i)

    if i > 1:
        SEEDS = [2732]
    plot_ppo_tom(DIR, run_name, SEEDS)