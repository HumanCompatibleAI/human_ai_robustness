
from human_aware_rl.ppo.ppo_pop import play_validation_games, make_validation_population

"""Quick code to take a trained ppo agent and play it with the validation set in ppo_pop"""

# Load agent:
seed = 2732
model_dir = 'val_pop_expt_cc0/cc_1_mtom'
base_dir = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/'
dir = base_dir + model_dir + '/'
from human_aware_rl.ppo.ppo_pop import get_ppo_agent
ppo_agent_to_check, _ = get_ppo_agent(dir, seed, best="val")


