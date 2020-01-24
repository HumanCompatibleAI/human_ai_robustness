"""File to generate bash scripts to run human_ai_robustness experiments"""
import os
from human_aware_rl import PROJECT_DIR
import numpy as np

# PARAMS

ppo_timesteps_l = {
    "croom": 1e7,
    # "aa": 1.6e7,
    # "cring": 1.6e7,
    # "fc": 1.6e7,
    # "cc": 1.6e7
}

shaping_horizon_l = {
    "croom": 1e6,
    # "aa": 6e6,
    # "cring": 5e6,
    # "fc": 5e6,
    # "cc": 5e6
}

minibatches_l = {
    "croom": 10,
    # "aa": 12,
    # "cring": 15,
    # "fc": 15,
    # "cc": 15
}

lr_annealing_l = {
    "croom": 3,
    # "aa": 3,
    # "cring": 1.5,
    # "fc": 1.5,
    # "cc": 1.5
}

sp_horizon_l = {
    "croom": [5e5,3e6],
    # "aa": [1e6, 7e6],
    # "cring": [2e6, 6e6],
    # "fc": [2e6, 6e6],
    # "cc": [2e6, 6e6]
}

# Default params at the time of writing:
# sim_threads = 30
# TOTAL_BATCH_SIZE = 12000
# MINIBATCHES = 6
# NUM_EVAL_GAMES = 100
# STEPS_PER_UPDATE = 8
# ENTROPY = 0.1
# VF_COEF = 0.1
# GAMMA = 0.99
# LAM = 0.98
# MAX_GRAD_NORM = 0.1
# CLIPPING = 0.05
# NUM_HIDDEN_LAYERS = 3
# SIZE_HIDDEN_LAYERS = 64
# NUM_FILTERS = 25
# NUM_CONV_LAYERS = 3

#------ functions ------#

def layout_codes(layout_name):
    if layout_name == 'cramped_room':
        return 'croom'

def make_command(gpu_id, ex_name, layout_code, layout_name, lr, seed, additional_params, testing=False):
    """Returns a bash command"""
    # if testing:
    #     return 'CUDA_VISIBLE_DEVICES={} python ppo/ppo.py with EX_NAME="{}" layout_name="{}" LR={:1.2e} ' \
    #            'OTHER_AGENT_TYPE="{}" SEEDS="{}" SELF_PLAY_HORIZON="[0, 1200]" TIMESTAMP_DIR=False ' \
               # 'ENVIRONMENT_TYPE="Overcooked" {} WAIT_TIME={} LOCAL_TESTING=True\n'.format(
            # gpu_id, ex_name, layout_name, lr, other_agent_type, seeds, extras, sleep_time
        # )

    return 'CUDA_VISIBLE_DEVICES={} python ppo/ppo_tom.py with EX_NAME="{}" layout_name="{}" REW_SHAPING_HORIZON={:1.2e} ' \
           'PPO_RUN_TOT_TIMESTEPS={:1.2e} LR={:1.2e} SEEDS=[{}] MINIBATCHES={} LR_ANNEALING={} ' \
           'SELF_PLAY_HORIZON={}'.format(
            gpu_id, ex_name, layout_name, shaping_horizon_l[layout_code], ppo_timesteps_l[layout_code], lr, seed,
            minibatches_l[layout_code], lr_annealing_l[layout_code], sp_horizon_l[layout_code]) + additional_params + '\n'

def flush_command_buffer(commands, runs_count, experiments_dir):
    filename = "experiments_{}.sh".format(runs_count)
    print("Flushing to", filename)
    f = open(os.path.join(experiments_dir, filename), "w")
    f.write(commands)
    f.close()
    return ""

#------- main -------#

np.random.seed(0)

num_seeds = 2

# Extra params to loop over:
param_to_loop = "TOTAL_BATCH_SIZE"  #TODO: Turn into a dictionary
param_to_loop_values = [12000, 14000, 16000, 18000]

layout_names = ['cramped_room']

lr = 1e-3

NUM_GPUS = 4
# seed_split = True
run_per_file = 64

# sleep_offset = 20
# sleep_index_reset = 32

testing = False

experiments_dir = PROJECT_DIR

commands = ""
runs_count = 0
sleep_time = 0

for layout_name in layout_names:
    layout_code = layout_codes(layout_name)
    curr_seeds = list(np.random.randint(0, 10000, size=num_seeds))

    for seed in curr_seeds:

        #TODO: Make two helper functions, one for extra params and one without them

        if param_to_loop:

            for i, param_to_loop_value in enumerate(param_to_loop_values):

                gpu_id = runs_count % NUM_GPUS + 1
                ex_name = '{}_s{}_{}'.format(layout_code, seed, i)

                additional_params = ' {}={}'.format(param_to_loop, param_to_loop_value)

                # If param to be looped over is the batch size then we need to adjust the sim_threads accordingly
                if param_to_loop == 'TOTAL_BATCH_SIZE':
                    sim_threads = param_to_loop_value // 400  # 400 is the horizon
                    additional_params += ' sim_threads={}'.format(sim_threads)

                commands += make_command(gpu_id, ex_name, layout_code, layout_name, lr, seed, additional_params,
                                         testing=testing)
                runs_count += 1
                # sleep_time = (runs_count % sleep_index_reset) * sleep_offset

            if runs_count % run_per_file == 0:
                commands = flush_command_buffer(commands, runs_count, experiments_dir)

flush_command_buffer(commands, runs_count, experiments_dir)