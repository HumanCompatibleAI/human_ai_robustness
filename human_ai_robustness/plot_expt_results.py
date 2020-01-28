
from human_aware_rl.ppo.ppo_tom import plot_ppo_tom

SEEDS = [2732, 9845]

DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/final_tune_results/ppo_runs'

for i in range(5):

    run_name = 'croom_' + str(i)

    plot_ppo_tom(DIR, run_name, SEEDS)