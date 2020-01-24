import os
import numpy as np
from itertools import product
from human_ai_coord.human_aware_rl import HUMAN_AWARE_RL_DIR
from human_model_theory.experiments.common import LAYOUT_NAMES, LAYOUT_NAMES_TO_CODES

# PARAMS

ppo_timesteps_l = {
    "croom": 8e6,
    "aa": 1.6e7,
    "cring": 1.6e7,
    "fc": 1.6e7,
    "cc": 1.6e7
}

shaping_horizon_l = {
    "croom": 1e6,
    "aa": 6e6,
    "cring": 5e6,
    "fc": 5e6,
    "cc": 5e6
}

minibatches_l = {
    "croom": 10,
    "aa": 12,
    "cring": 15,
    "fc": 15,
    "cc": 15
}

lr_annealing_l = {
    "croom": 3,
    "aa": 3,
    "cring": 1.5,
    "fc": 1.5,
    "cc": 1.5
}

sp_horizon_l = {
    "croom": [5e5, 3e6],
    "aa": [1e6, 7e6],
    "cring": [2e6, 6e6],
    "fc": [2e6, 6e6],
    "cc": [2e6, 6e6]
}


def make_command(gpu_id, run_name, layout_code, layout_name, lr, other_agent_type, seeds, extras, sleep_time,
                 testing=False):
    if testing:
        return 'CUDA_VISIBLE_DEVICES={} python ppo/ppo.py with EX_NAME="{}" layout_name="{}" LR={:1.2e} OTHER_AGENT_TYPE="{}" SEEDS="{}" SELF_PLAY_HORIZON="[0, 1200]" TIMESTAMP_DIR=False ENVIRONMENT_TYPE="Overcooked" {} WAIT_TIME={} LOCAL_TESTING=True\n'.format(
            gpu_id, run_name, layout_name, lr, other_agent_type, seeds, extras, sleep_time
        )

    return 'CUDA_VISIBLE_DEVICES={} python ppo/ppo.py with EX_NAME="{}" layout_name="{}" REW_SHAPING_HORIZON={:1.2e} PPO_RUN_TOT_TIMESTEPS={:1.2e} LR={:1.2e} OTHER_AGENT_TYPE="{}" SEEDS="{}" MINIBATCHES={} LR_ANNEALING={} SELF_PLAY_HORIZON="{}" TIMESTAMP_DIR=False ENVIRONMENT_TYPE="Overcooked" {} WAIT_TIME={}\n'.format(
        gpu_id, run_name, layout_name, shaping_horizon_l[layout_code], ppo_timesteps_l[layout_code], lr,
        other_agent_type, seeds, minibatches_l[layout_code], lr_annealing_l[layout_code], sp_horizon_l[layout_code],
        extras, sleep_time
    )


def make_evals_command(curr_seeds, eval_num_games, eval_thresholds):
    eval_command = " eval_seeds=\"{}\" eval_num_games={}".format(str(curr_seeds), eval_num_games)
    if eval_thresholds is not None:
        eval_command += " eval_thresholds=\"{}\"".format(eval_thresholds)
    return eval_command


def get_eval_threshs_and_run_name(run_type, run_threshold, layout_code, percentiles):
    if run_type in ["opt", "rnd", "bad"]:
        other_agent_type = "bc_{}".format(run_type)
        # For threshold-conditional rollouts, set the classifier threshold
        extras = "OTHER_AGENT_KWARGS=\"{\'CLASSIFIER_THRESH\': " + str(run_threshold) + " }\""
        run_name = "bc_{}{}_{}".format(run_type, run_threshold, layout_code)
        eval_threshs = [run_threshold]
    elif run_type in ["sp", "bc"]:
        other_agent_type = "sp" if run_type == "sp" else "bc_base"
        # The run doesn't actually have a threshold, but has to eval on all the thresholds
        eval_threshs = percentiles
        run_name = "{}_base_{}".format(run_type, layout_code)
        extras = ""
    else:
        raise ValueError()
    return eval_threshs, run_name, other_agent_type, extras


def flush_command_buffer(commands, runs_count, experiments_dir):
    filename = "experiments_{}.sh".format(runs_count)
    print("Flushing to", filename)
    f = open(os.path.join(experiments_dir, filename), "w")
    f.write(commands)
    f.close()
    return ""


np.random.seed(0)

lr = 1e-3

layouts = [
    "asymmetric_advantages"]  # ["cramped_room"]#, "asymmetric_advantages", "coordination_ring", "counter_circuit"] #"forced_coordination"
threshold_run_types = ["opt", "rnd"]  # "bad"
percentiles = [0, 25, 50, 80, 90, 95, 98, 99, 100, 125, 150]  # [0, 25, 50, 80, 90, 95, 98, 99, 100, 125, 1000]
run_types = list(product(threshold_run_types, percentiles)) + [("bc", "")] + [("sp", "")]
NUM_GPUS = 4
seed_split = True
num_seeds = 5
run_per_file = 64

sleep_offset = 20
sleep_index_reset = 32

testing = False
eval_num_games = 40

experiments_dir = os.path.join(HUMAN_AWARE_RL_DIR, "ex")

print("{} run types: {}".format(len(run_types), run_types))

commands = ""
runs_count = 0
sleep_time = 0
for layout_name in layouts:

    layout_code = LAYOUT_NAMES_TO_CODES[layout_name]

    for run_type, run_threshold in run_types:
        gpu_id = runs_count % NUM_GPUS + 1
        curr_seeds = list(np.random.randint(0, 10000, size=num_seeds))
        if not seed_split:
            curr_seeds = [curr_seeds]

        eval_threshs, run_name, other_agent_type, extras = get_eval_threshs_and_run_name(run_type, run_threshold,
                                                                                         layout_code, percentiles)
        print(run_name)

        for seed in curr_seeds:
            seed_list = seed if type(seed) is list else [seed]
            assert type(seed_list[0] is int)

            if seed == curr_seeds[-1]:
                # For last seed, do evals
                extras += make_evals_command(curr_seeds, eval_num_games, eval_threshs)

            commands += make_command(gpu_id, run_name, layout_code, layout_name, lr, other_agent_type, seed_list,
                                     extras, sleep_time, testing=testing)
            runs_count += 1
            sleep_time = (runs_count % sleep_index_reset) * sleep_offset

            if runs_count % run_per_file == 0:
                commands = flush_command_buffer(commands, runs_count, experiments_dir)

flush_command_buffer(commands, runs_count, experiments_dir)