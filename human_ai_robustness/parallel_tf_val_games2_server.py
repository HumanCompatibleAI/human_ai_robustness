import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent
import time, random, copy
import numpy as np
from human_aware_rl.baselines_utils import get_vectorized_gym_env
from overcooked_ai_py.agents.agent import AgentPair, AsymmAgentPairs
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, MultiOvercookedEnv
from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
from human_aware_rl.ppo.ppo_pop import make_tom_agent
from human_ai_robustness.import_person_params import import_manual_tom_params
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from human_aware_rl.data_dir import DATA_DIR

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

# def play_single_validation_game(parallel_env):
#     agent_pair = parallel_env.agent_pair
#     num_val_games = parallel_env.params["NUM_VAL_GAMES"]
#     trajs = parallel_env.get_rollouts(agent_pair, num_games=num_val_games, final_state=False, display=False)
#     sparse_rews = trajs["ep_returns"]
#     avg_sparse_rew = np.mean(sparse_rews)
#     return avg_sparse_rew

def imitate_play_validation_games(params, ppo_agent0, ppo_agent1, parallel_envs, mdp, multi_env, val_pop, rearranged_val_pop, rearranged_val_pop_num_avg):

    # Need a different ppo agent for each index:
    ppo_agent0.set_mdp(mdp)
    ppo_agent1.set_mdp(mdp)

    # Set up the agent pairs and put them in the parallel envs:
    for i, parallel_env in enumerate(parallel_envs):
        agent_pair = AgentPair(ppo_agent0, parallel_env.val_agent) if i % 2 == 0 else AgentPair(parallel_env.val_agent, ppo_agent1)
        parallel_env.agent_pair = agent_pair
        parallel_env.params = params

    # Manual loop:
    time0 = time.perf_counter()
    # np.random.seed(0)
    # random.seed(0)
    validation_rewards = []
    for parallel_env in parallel_envs:
        agent_pair = parallel_env.agent_pair
        trajs = parallel_env.get_rollouts(agent_pair, num_games=params["NUM_VAL_GAMES"],
                                            final_state=False, display=False)
        sparse_rews = trajs["ep_returns"]
        avg_sparse_rew = np.mean(sparse_rews)
        validation_rewards.append(avg_sparse_rew)
    time_loop = time.perf_counter() - time0


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#
    # Using MultiOvercookedEnv and AsymmAgentPairs:
    asymm_agent_pairs = []
    for i in range(len(rearranged_val_pop)):
        ppo_agent_indices = []
        for j in range(multi_env.num_envs):
            ppo_agent_indices.append(0 if j % 2 == 0 else 1)
            #TODO: Don't need to do this here??:
            rearranged_val_pop[i][j].set_agent_index(1 - ppo_agent_indices[j])
        asymm_agent_pairs.append(AsymmAgentPairs(ppo_agent0, rearranged_val_pop[i], single_agent_indices=ppo_agent_indices))

    [asymm_agent_pairs[i].reset() for i in range(len(asymm_agent_pairs))]

    time1 = time.perf_counter()
    # np.random.seed(0)
    # random.seed(0)
    validation_rewards_partf = []
    for _ in range(rearranged_val_pop_num_avg):
        for i in range(len(rearranged_val_pop)):
            trajs = multi_env.get_asymm_rollouts(asymm_agent_pairs[i], num_games=1)
            assert len(trajs) == multi_env.num_envs
            for j in range(multi_env.num_envs):
                sparse_rews = trajs[j]["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                validation_rewards_partf.append(avg_sparse_rew)

    time_partf = time.perf_counter() - time1

    print("Series rews: ", validation_rewards, "\nPar-tf rews: ", validation_rewards_partf)
    print("Series rews = ", np.mean(validation_rewards), "\nParall rews = ", np.mean(validation_rewards_partf))
    print("Loop took: {}, parallel took: {}".format(time_loop, time_partf))


def sort_into_rearranged_pop(agent, ppo_sim_threads, count):
    if ppo_sim_threads == 30:
        if count < 30:
            rearranged_val_pop[0].append(agent)
        elif 30 <= count < 60:
            rearranged_val_pop[1].append(agent)
        elif 60 <= count < 90:
            rearranged_val_pop[2].append(agent)
        elif 90 <= count < 120:
            rearranged_val_pop[3].append(agent)
        else:
            raise ValueError('Wrong population / average sizes')
    elif ppo_sim_threads == 60:
        if count < 60:
            rearranged_val_pop[0].append(agent)
        elif 60 <= count < 120:
            rearranged_val_pop[1].append(agent)
        else:
            raise ValueError('Wrong population / average sizes')
    return rearranged_val_pop

##################
# PARAMS #
##################
layout_name = "counter_circuit"
sim_threads = 4
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NUM_VAL_GAMES = 90
VAL_POP_SIZE = 20
#<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
DISPLAY_VAL_GAMES = False
start_order_list = ['any']*100
rew_shaping_params = {}
horizon = 400
mdp_generation_params = {}
params = {
    "mdp_params": {
        "layout_name": layout_name,
        "start_order_list": start_order_list,
        "rew_shaping_params": rew_shaping_params
    },
    "env_params": {
        "horizon": horizon
    },
    "sim_threads": sim_threads,
    "NUM_VAL_GAMES": NUM_VAL_GAMES,
    "DISPLAY_VAL_GAMES": DISPLAY_VAL_GAMES,
    "VAL_POP_SIZE": VAL_POP_SIZE,
}

# Pick an arbitrary ppo model and make an agent:
base_dir = '/home/paul/agents_to_QT/'
model_dir = 'val_expt_aa_cc1/cc_20_mantoms'
seed = 2732

dir = base_dir + model_dir + '/'
from human_aware_rl.ppo.ppo_pop import get_ppo_agent
ppo_agent0, config0 = get_ppo_agent(dir, seed, best='train')

time0 = time.perf_counter()
ppo_agent1, config1 = get_ppo_agent(dir, seed, best='train')
print('Time make ppo: ', time.perf_counter()-time0)
time0 = time.perf_counter()
ppo_agent_copy = copy.deepcopy(ppo_agent1)
print('Time copy ppo: ', time.perf_counter()-time0)

# Make the standard mdp for this layout:
layout = params["mdp_params"]["layout_name"]
mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=['any'] * 100, cook_time=20, rew_shaping_params=None)
no_counters_params['counter_drop'] = mdp.get_counter_locations()
no_counters_params['counter_goals'] = mdp.get_counter_locations()
mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

assert NUM_VAL_GAMES % 3 == 0, "Only set up for number of val games is a multiple of 3!"
assert config0['sim_threads'] == 30 or config0['sim_threads'] == 60, "Only set up for ppo agents with 30 or 60 threads"
assert VAL_POP_SIZE == 20, "Only set up for val pop size = 20"

parallel_envs = []
val_pop = []
rearranged_val_pop = [[], [], [], []] if config0['sim_threads'] == 30 else [[], []]
VAL_TOM_PARAMS, _, _ = import_manual_tom_params(layout, 20)
count=0
for _ in range(3):  # x3 because we have a population of 20*2, and need 120 agents in order to split them into sub-groups of size 30 or 60 (the ppo's sim_threads)
    for i in range(VAL_POP_SIZE):
        for j in range(2):  # One for each index
            parallel_env = OvercookedEnv(mdp, **params["env_params"])
            tom_agent = make_tom_agent(mlp)

            #>>> Override as we only have 10 val toms but we want a pop of 20 for testing!
            if i >= 10:
                i -= 10
            #>>>

            tom_agent.set_tom_params(None, None, VAL_TOM_PARAMS, tom_params_choice=i)

            parallel_env.val_agent = tom_agent
            if count < 40:
                parallel_envs.append(parallel_env)
            val_pop.append(tom_agent)
            rearranged_val_pop = sort_into_rearranged_pop(tom_agent, config0['sim_threads'], count)
            count += 1

rearranged_val_pop_num_avg = int(NUM_VAL_GAMES / 3)
assert len(rearranged_val_pop)*config0['sim_threads'] == count
#
# # Set up parallel environments with validation population in each:
# parallel_envs = []
# val_pop = []
# VAL_TOM_PARAMS, _, _ = import_manual_tom_params(layout, 20)
# for i in range(VAL_POP_SIZE):
#     for j in range(2):  # One for each index
#         parallel_env = OvercookedEnv(mdp, **params["env_params"])
#         tom_agent = make_tom_agent(mlp)
#         tom_agent.set_tom_params(None, None, VAL_TOM_PARAMS, tom_params_choice=i)
#         parallel_env.val_agent = tom_agent
#         parallel_envs.append(parallel_env)
#         val_pop.append(tom_agent)

# Set up MultiEnvs:
multi_env = MultiOvercookedEnv(config0['sim_threads'], mdp, **params["env_params"])

imitate_play_validation_games(params, ppo_agent0, ppo_agent1, parallel_envs, mdp, multi_env, val_pop, rearranged_val_pop, rearranged_val_pop_num_avg)

print('')