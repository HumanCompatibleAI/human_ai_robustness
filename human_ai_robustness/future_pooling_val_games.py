
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

def play_single_validation_game(agent_pair_and_other_stuff):
    agent_pair, other_stuff = agent_pair_and_other_stuff
    overcooked_env = copy.copy(other_stuff["overcooked_env"])
    trajs = overcooked_env.get_rollouts(agent_pair, num_games=other_stuff["params"]["NUM_VAL_GAMES"],
                                        final_state=False, display=False)
    sparse_rews = trajs["ep_returns"]
    avg_sparse_rew = np.mean(sparse_rews)
    return avg_sparse_rew

def imitate_play_validation_games(params, ppo_agent, validation_pop, mdp):

    ppo_agent.set_mdp(mdp)
    overcooked_env = OvercookedEnv(mdp, **params["env_params"])

    # Set up the agent pairs:
    agent_pairs = []
    for val_agent in validation_pop:
        for ppo_index in range(2):
            agent_pair = AgentPair(ppo_agent, val_agent) if ppo_index == 0 else AgentPair(val_agent, ppo_agent)
            agent_pairs.append(agent_pair)

    time0 = time.perf_counter()
    # Manual loop:
    np.random.seed(0)
    random.seed(0)
    validation_rewards = []
    for agent_pair in agent_pairs:
        trajs = overcooked_env.get_rollouts(agent_pair, num_games=params["NUM_VAL_GAMES"],
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
    other_stuff = {"overcooked_env": overcooked_env, "params": params}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        validation_rewards_parallel = list(executor.map(play_single_validation_game, [[agent_pairs[k], other_stuff] for k in range(len(agent_pairs))]))
    time_par = time.perf_counter() - time1

    print("Series rews: ", np.mean(validation_rewards), "\nParall rews: ", np.mean(validation_rewards_parallel))
    print("Loop took: {}, parallel took: {}".format(time_loop, time_par))

    mean_val_rews = np.mean(validation_rewards)

##################
# PARAMS #
##################
layout_name = "counter_circuit"
sim_threads = 4
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
NUM_VAL_GAMES = 120
VAL_POP_SIZE = 4
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
base_dir = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/'
model_dir = 'bc_pop_cc/cc_1_bc_p'
seed = 2732

dir = base_dir + model_dir + '/'
from human_aware_rl.ppo.ppo_pop import get_ppo_agent
ppo_agent, _ = get_ppo_agent(dir, seed, best=True)

# Make the standard mdp for this layout:
layout = params["mdp_params"]["layout_name"]
mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=['any'] * 100, cook_time=20, rew_shaping_params=None)
no_counters_params['counter_drop'] = mdp.get_counter_locations()
no_counters_params['counter_goals'] = mdp.get_counter_locations()
mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

# Make a fake validation pop:
validation_population = []
# Make TOM pop
VAL_TOM_PARAMS, _, _ = import_manual_tom_params(layout, 20)
for i in range(VAL_POP_SIZE):
    tom_agent = make_tom_agent(mlp)
    tom_agent.set_tom_params(None, None, VAL_TOM_PARAMS, tom_params_choice=i)
    validation_population.append(tom_agent)

imitate_play_validation_games(params, ppo_agent, validation_population, mdp)

print('')