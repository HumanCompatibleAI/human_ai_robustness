import logging
from concurrent.futures import ThreadPoolExecutor
import concurrent
import time, random, copy
import numpy as np
from human_aware_rl.baselines_utils import get_vectorized_gym_env
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
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

def play_single_validation_game(parallel_env):
    agent_pair = parallel_env.agent_pair
    num_val_games = parallel_env.params["NUM_VAL_GAMES"]
    trajs = parallel_env.get_rollouts(agent_pair, num_games=num_val_games, final_state=False, display=False)
    sparse_rews = trajs["ep_returns"]
    avg_sparse_rew = np.mean(sparse_rews)
    return avg_sparse_rew

def imitate_play_validation_games(params, ppo_agent0, ppo_agent1, parallel_envs, mdp):

    # Need a different ppo agent for each index:
    ppo_agent0.set_mdp(mdp)
    ppo_agent1.set_mdp(mdp)

    # Set up the agent pairs and put them in the parallel envs:
    for i, parallel_env in enumerate(parallel_envs):
        agent_pair = AgentPair(ppo_agent0, parallel_env.val_agent) if i % 2 == 0 else AgentPair(parallel_env.val_agent, ppo_agent1)
        parallel_env.agent_pair = agent_pair
        parallel_env.params = params

    time0 = time.perf_counter()
    # Manual loop:
    np.random.seed(0)
    random.seed(0)
    validation_rewards = []
    for parallel_env in parallel_envs:
        agent_pair = parallel_env.agent_pair
        trajs = parallel_env.get_rollouts(agent_pair, num_games=params["NUM_VAL_GAMES"],
                                            final_state=False, display=False)
        sparse_rews = trajs["ep_returns"]
        avg_sparse_rew = np.mean(sparse_rews)
        validation_rewards.append(avg_sparse_rew)
    time_loop = time.perf_counter() - time0

    # Manual loop 2, using play_single_validation_game:
    # np.random.seed(0)
    # random.seed(0)
    # validation_rewards2 = []
    # other_stuff = {"overcooked_env": overcooked_env, "params": params}
    # for agent_pair in agent_pairs:
    #     validation_rewards2.append(play_single_validation_game([agent_pair, other_stuff]))

    time1 = time.perf_counter()
    # Parallel loop:
    np.random.seed(0)
    random.seed(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=params["sim_threads"]) as executor:
        validation_rewards_parallel = list(executor.map(play_single_validation_game, parallel_envs))
    time_par = time.perf_counter() - time1

    print("Series rews: ", validation_rewards, "\nParall rews: ", validation_rewards_parallel)
    print("Series rews = ", np.mean(validation_rewards), "\nParall rews = ", np.mean(validation_rewards_parallel))
    print("Loop took: {}, parallel took: {}".format(time_loop, time_par))

    mean_val_rews = np.mean(validation_rewards)

##################
# PARAMS #
##################
layout_name = "counter_circuit"
sim_threads = 20
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NUM_VAL_GAMES = 100
VAL_POP_SIZE = 10
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
ppo_agent0, _ = get_ppo_agent(dir, seed, best='train')
ppo_agent1, _ = get_ppo_agent(dir, seed, best='train')

# Make the standard mdp for this layout:
layout = params["mdp_params"]["layout_name"]
mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=['any'] * 100, cook_time=20, rew_shaping_params=None)
no_counters_params['counter_drop'] = mdp.get_counter_locations()
no_counters_params['counter_goals'] = mdp.get_counter_locations()
mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

# Set up parallel environments with validation population in each:
parallel_envs = []
VAL_TOM_PARAMS, _, _ = import_manual_tom_params(layout, 20)
for i in range(VAL_POP_SIZE):
    for j in range(2):  # One for each index
        parallel_env = OvercookedEnv(mdp, **params["env_params"])
        tom_agent = make_tom_agent(mlp)
        tom_agent.set_tom_params(None, None, VAL_TOM_PARAMS, tom_params_choice=i)
        parallel_env.val_agent = tom_agent
        parallel_envs.append(parallel_env)

# logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
# logging.getLogger().setLevel(logging.INFO)  # pk: Not sure why this line is also needed

imitate_play_validation_games(params, ppo_agent0, ppo_agent1, parallel_envs, mdp)

print('')