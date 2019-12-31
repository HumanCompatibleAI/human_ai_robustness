import os, gym, time, sys, random, itertools, logging
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver

# For sacred experiments -- NEEDS TO GO HERE, before importing other modules, otherwise sacred gives an error:
PBT_DATA_DIR = "data/pbt_hms_runs/"
ex = Experiment('PBT_HMS')
ex.observers.append(FileStorageObserver.create(PBT_DATA_DIR))

from overcooked_ai_py.utils import profile, load_pickle, save_pickle, save_dict_to_file, load_dict_from_file
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.agents.agent import AgentPair, ToMModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, \
    reset_tf, set_global_seed
from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env, \
    update_model, get_agent_from_model, save_baselines_model, overwrite_model, \
    load_baselines_model, LinearAnnealer, LinearAnnealerZeroToOne, delay_before_run
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
# from memory_profiler import profile
import tracemalloc as tm
import gc  # For garbage collection!

# Suppress warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This surpresses tf errors, mainly this one was getting in the way: (E
# tensorflow/core/common_runtime/bfc_allocator.cc:373] tried to deallocate nullptr)
os.environ['KMP_WARNINGS'] = 'off'  # This is meant to suppress the "OMP: Info #250: KMP_AFFINITY" error
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

# Start tracking memory. Store 25 frames (?!)
tm.start(10)

class MemSnaps(object):
    """For storing memory snapshots"""
    def __init__(self, curr_snap, prev_snap=None):
        self.curr_snap = curr_snap
        self.prev_snap = prev_snap


class HMAgent(object):
    """A human-model agent. This class should mimic the necessary parts of PPOAgent."""

    def __init__(self, start_params, hm_number, player_index=99, agent_name=None):
        self.params = start_params
        self.agent_name = agent_name
        self.player_index = player_index
        self.hm_number = hm_number  # Give each HM a unique number, so that the personality parameters can be extracted
        self.human_model = True
        # from params
        # self.logs = start_logs if start_logs is not None else {...}  # We don't need logs for the human models?

        # Personality parameters: (See agent.py for definitions)
        self.perseverance = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['PERSEVERANCE_HM{}'.format(hm_number)]
        self.teamwork = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['TEAMWORK_HM{}'.format(hm_number)]
        self.retain_goals = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['RETAIN_GOALS_HM{}'.format(hm_number)]
        self.wrong_decisions = self.params['PERSON_PARAMS_HM{}'.format(hm_number)][
            'WRONG_DECISIONS_HM{}'.format(hm_number)]
        self.thinking_prob = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['THINKING_PROB_HM{}'.format(hm_number)]
        self.path_teamwork = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['PATH_TEAMWORK_HM{}'.format(hm_number)]
        self.rationality_coeff = self.params['PERSON_PARAMS_HM{}'.format(hm_number)][
            'RATIONALITY_COEFF_HM{}'.format(hm_number)]
        self.prob_pausing = self.params['PERSON_PARAMS_HM{}'.format(hm_number)]['PROB_PAUSING_HM{}'.format(hm_number)]

    def get_agent(self, mlp):
        return ToMModel(mlp=mlp, player_index=self.player_index, perseverance=self.perseverance,
                        teamwork=self.teamwork, retain_goals=self.retain_goals,
                        wrong_decisions=self.wrong_decisions, thinking_prob=self.thinking_prob,
                        path_teamwork=self.path_teamwork, rationality_coefficient=self.rationality_coeff,
                        prob_pausing=self.prob_pausing)

    def get_multi_agent(self, mlp):
        """Get sim_threads number of agents, for use in training the models"""
        multi_agent = []
        for i in range(self.params["sim_threads"]):
            multi_agent.append(self.get_agent(mlp))
        # Note that doing [self.get_agent(mlp)]*N creates N copies that share the same parameters, e.g. same self.agent_idx
        return multi_agent

class PPOAgent(object):
    """An agent that can be saved and loaded and all and the main data it contains is the self.model
    
    Goal is to be able to pass in save_locations or PPOAgents to workers that will load such agents
    and train them together.
    """
    
    def __init__(self, agent_name, start_params, combined_pop_size=1, start_logs=None, model=None, gym_env=None):
        # TODO: hacky init code
        self.params = start_params
        self.human_model = False
        self.logs = start_logs if start_logs is not None else {
            "agent_name": agent_name,
            "avg_rew_per_step": [],
            "params_hist": defaultdict(list),
            "num_ppo_runs": 0,
            "reward_shaping": [],
            "weight_hm": []
        }
        # Log to keep track of performance with each agent
        for i in range(combined_pop_size):
            log_name = "dense_rew_with_agent{}".format(i)
            self.logs[log_name] = []
            log_name = "sparse_rew_with_agent{}".format(i)
            self.logs[log_name] = []

        # For agent 0 only, record the best agent's sparse reward (Note: no reason the best agent is agent 0!)
        if agent_name == "ppo_agent0":
            self.logs["best_agent_rew_bc"] = []
            self.logs["best_agent_rew_hm0"] = []
            self.logs["best_agent_rew_hm1"] = []


        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            self.model = model if model is not None else create_model(gym_env, agent_name, **start_params)  # pk:
            # create the model

    @property
    def num_ppo_runs(self):
        return self.logs["num_ppo_runs"]
    
    @property
    def agent_name(self):
        return self.logs["agent_name"]

    def get_agent(self, mlp):
        return get_agent_from_model(self.model, self.params["sim_threads"])

    def get_multi_agent(self, mlp):
        return self.get_agent(mlp)  # The PPOAgent is always a "multi agent", so this function is the same as get_agent

    def update(self, gym_env):
        with tf.device('/device:GPU:{}'.format(self.params["GPU_ID"])):
            train_info = update_model(gym_env, self.model, **self.params)

            for k, v in train_info.items():
                if k not in self.logs.keys():
                    self.logs[k] = []
                self.logs[k].extend(v)
            self.logs["num_ppo_runs"] += 1

    def update_avg_rew_per_step_logs(self, avg_rew_per_step_stats):
        self.logs["avg_rew_per_step"] = avg_rew_per_step_stats

    def save(self, save_folder):
        """Save agent model, logs, and parameters"""
        create_dir_if_not_exists(save_folder)
        save_baselines_model(self.model, save_folder)
        save_dict_to_file(self.logs, save_folder + "logs") # New code has this instead: save_dict_to_file(dict(self.logs), save_folder + "logs")
        save_dict_to_file(self.params, save_folder + "params")

    @staticmethod
    def from_dir(load_folder):
        logs = load_dict_from_file(load_folder + "/logs.txt")
        agent_name = logs["agent_name"]
        params = load_dict_from_file(load_folder + "/params.txt")
        model = load_baselines_model(load_folder, agent_name, config=params)
        return PPOAgent(agent_name, params, start_logs=logs, model=model)

    @staticmethod
    def update_from_files(file0, file1, gym_env, save_dir):
        pbt_agent0 = PPOAgent.from_dir(file0)
        pbt_agent1 = PPOAgent.from_dir(file1)
        gym_env.other_agent = pbt_agent1
        pbt_agent0.update(gym_env)
        return pbt_agent0

    def save_predictor(self, save_folder):
        """Saves easy-to-load simple_save tensorflow predictor for agent"""
        simple_save(
            tf.get_default_session(),
            save_folder,
            inputs={"obs": self.model.act_model.X},
            outputs={
                "action": self.model.act_model.action,
                "value": self.model.act_model.vf,
                "action_probs": self.model.act_model.action_probs
            })

    def update_pbt_iter_logs(self):
        for k, v in self.params.items():
            self.logs["params_hist"][k].append(v)
        self.logs["params_hist"] = dict(self.logs["params_hist"])

    def explore_from(self, best_training_agent):
        overwrite_model(best_training_agent.model, self.model)
        self.logs["num_ppo_runs"] = best_training_agent.num_ppo_runs
        self.params = self.mutate_params(best_training_agent.params)

    def mutate_params(self, params_to_mutate):
        params_to_mutate = params_to_mutate.copy()
        for k in self.params["HYPERPARAMS_TO_MUTATE"]:
            if np.random.random() < params_to_mutate["RESAMPLE_PROB"]:
                mutation = np.random.choice(self.params["MUTATION_FACTORS"])
                
                if k == "LAM": 
                    # Move eps/2 in either direction
                    eps = min(
                        (1 - params_to_mutate[k]) / 2,      # If lam is > 0.5, avoid going over 1
                        params_to_mutate[k] / 2             # If lam is < 0.5, avoid going under 0
                    )
                    rnd_direction = (-1)**np.random.randint(2) 
                    mutation = rnd_direction * eps
                    params_to_mutate[k] = params_to_mutate[k] + mutation
                elif type(params_to_mutate[k]) is int:
                    params_to_mutate[k] = max(int(params_to_mutate[k] * mutation), 1)
                else:
                    params_to_mutate[k] = params_to_mutate[k] * mutation
                    
                print("Mutated {} by a factor of {}".format(k, mutation))

        print("Old params", self.params)
        print("New params", params_to_mutate)
        return params_to_mutate

#pk: Most defaults kept the same as pbt.py from branch pk-dev... Some params commented out if it's not clear what they
# do / if I need them
@ex.config
def my_config():
    LOCAL_TESTING = False

    GPU_ID = 0
    RUN_TYPE = "pbt"  #pk: Needed for making gym envs

    # PARAMS
    EX_DIR = "unnamed"
    EX_NAME = "unnamed"

    TIMESTAMP_DIR = False

    if TIMESTAMP_DIR:
        SAVE_DIR = PBT_DATA_DIR + EX_DIR + "/"+ time.strftime('%Y_%m_%d-%H_%M_%S') + "/"
    else:
        SAVE_DIR = PBT_DATA_DIR + EX_DIR + "/" + EX_NAME + "/"

    logging.info("Saving data to ", SAVE_DIR)
    MODEL_SAVE_FREQUENCY = 10

    DELAY_MINS = None
    FORCE_KILL = True

    # PBT params
    sim_threads = 50 if not LOCAL_TESTING else 4
    PPO_POP_SIZE = 2 if not LOCAL_TESTING else 1
    HM_POP_SIZE = 2 if not LOCAL_TESTING else 1
    COMBINED_POP_SIZE = PPO_POP_SIZE + HM_POP_SIZE
    # ITER_PER_SELECTION = COMBINED_POP_SIZE * PPO_POP_SIZE  # How many pairings and model training updates before the
    # worst model is overwritten.
    PPO_RUN_TOT_TIMESTEPS = 40000 if not LOCAL_TESTING else 100  #pk: How any timesteps each time we "update" the
    # model using .update function (?)

    #---------------#
    # Work out how many PBT ITERs:
    TOTAL_STEPS_PER_AGENT = 5e6 if not LOCAL_TESTING else 4e2  # MIN steps before stop training (actual number varies)
    INCLUDE_HMS_AFTER = 3e6 if not LOCAL_TESTING else -1  # MIN steps before including HM
    actual_iters_pre_hms = int(np.ceil(INCLUDE_HMS_AFTER / (PPO_POP_SIZE * PPO_RUN_TOT_TIMESTEPS)))
    actual_steps_pre_hms = actual_iters_pre_hms * PPO_POP_SIZE * PPO_RUN_TOT_TIMESTEPS
    NUM_PBT_ITER = actual_iters_pre_hms \
        + int(np.ceil((TOTAL_STEPS_PER_AGENT - actual_steps_pre_hms) / (COMBINED_POP_SIZE * PPO_RUN_TOT_TIMESTEPS)))
    # = timesteps without hm / steps per iteration wo hms + timesteps with hm / steps per iter with hm
    # Given how many timesteps we want (TOTAL_STEPS_PER_AGENT), this determines how many pbt_iters we should do (
    # following how many times ".update" is called shows how we arrive at this number for NUM_PBT_ITER)
    #---------------#

    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 100  # Local testing reduced
    ENTROPY = 0.5
    GAMMA = 0.99  # Can try up to 0.998
    MAX_GRAD_NORM = 0.1
    LR = 1e-3  # 1e-3 max
    VF_COEF = 0.5  # Try up to 0.5
    STEPS_PER_UPDATE = 8 if not LOCAL_TESTING else 1
    MINIBATCHES = 5 if not LOCAL_TESTING else 1
    CLIPPING = 0.05
    LAM = 0.98
    RESAMPLE_PROB = 0.33
    MUTATION_FACTORS = [0.75, 1.25]
    HYPERPARAMS_TO_MUTATE = ["LAM", "CLIPPING", "LR", "STEPS_PER_UPDATE", "ENTROPY", "VF_COEF"]
    NETWORK_TYPE = "conv_and_mlp"

    DISPLAY_TRAINING = False  # Display the overcooke game (for 1 random SIM_THREAD of the HM) during training

    # Weight the hm agents differently when assessing performance of agents. This gives the maximum weighting given to
    #the human models, compared to the ppos, during selection in pbt. We anneal the weight from 0 to this max value.
    WEIGHT_HM_MAX = 1

    # Mdp params
    FIXED_MDP = True
    LAYOUT_NAME = "simple"  # BC automatically changes when layout changes
    horizon = 400 if not LOCAL_TESTING else 10
    START_ORDER_LIST = ["any"] * 10
    RND_OBJS = 0.0
    RND_POS = False
    REW_SHAPING_HORIZON = 1e6
    # WEIGHT_HM_HORIZON = 5e6  # Around 5e6 env steps the PPO SEEMS to not get much better (??????)

    # For non fixed MDPs
    PADDED_MDP_SHAPE = (11, 7)
    MDP_SHAPE_FN = ([5, 11], [5, 7])
    PROP_EMPTY_FN = [0.6, 1]
    PROP_FEATS_FN = [0, 0.6]

    REW_SHAPING_PARAMS = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }

    NUM_HIDDEN_LAYERS = 1  # Try 2
    SIZE_HIDDEN_LAYERS = 32  # Try 64
    NUM_FILTERS = 15  # Try 25
    NUM_CONV_LAYERS = 3 if not LOCAL_TESTING else 1

    SEEDS = [0]
    NUM_SELECTION_GAMES = 10 if not LOCAL_TESTING else 1
    NUM_EVAL_GAMES = 5 if not LOCAL_TESTING else 1 # Number of games used when evaluating the best_agent in each iteration

    # NO_COUNTER_PARAMS:
    START_ORIENTATIONS = False
    WAIT_ALLOWED = False
    COUNTER_PICKUP = []
    SAME_MOTION_GOALS = True

    #--------------------------------------------#
    bc_seed = '2'  # Micah uses different seeds for different layouts: "simple": [2, 0], "unident_s": [1, 1],
    # "random1": [0, 1], "random0": [4, 1], "random3": [1, 3]. <-- in layout [train, test]
    bc_train_or_test = 'train'
    BC_AGENT_FOLDER = '{}_bc_{}_seed{}'.format(LAYOUT_NAME, bc_train_or_test, bc_seed)  # BC model to test with

    # Personality parameters for the human models:
    # HM0: Best values from 1st attempt to fit params to human data, on layout Simple. Run "Opt2". Params were
    # initialised at 0.5.
    PROB_PAUSING = 0.7238080301129235
    PERSON_PARAMS_HM0 = {'PERSEVERANCE_HM0': 0.7127657165010315, 'TEAMWORK_HM0': 0.5761362547930812, 'RETAIN_GOALS_HM0':
            0.4120858711447392, 'WRONG_DECISIONS_HM0': 0.17599408591437354, 'THINKING_PROB_HM0': 1, 'PATH_TEAMWORK_HM0':
            0.5187008050023094, 'RATIONALITY_COEFF_HM0': 9.301698992594565, 'PROB_PAUSING_HM0': PROB_PAUSING}

    # HM1: Params initialised at 1 (different final params, but with similar loss, as compared to HM0)
    PERSON_PARAMS_HM1 = {'PERSEVERANCE_HM1': 0.9716476326036333, 'TEAMWORK_HM1': 0.943273339206386, 'RETAIN_GOALS_HM1':
        0.5860196577665008, 'WRONG_DECISIONS_HM1': 0.17517911236529413, 'THINKING_PROB_HM1': 1, 'PATH_TEAMWORK_HM1':
        0.901180284069246, 'RATIONALITY_COEFF_HM1': 20.67380327560989, 'PROB_PAUSING_HM1': PROB_PAUSING}

    # Old HM1: Reasonably greedy
    # Old HM2: Highly rational and teamworky
    # Old HM3: Erratic, irrational, sub-optimal!
    #--------------------------------------#


    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    # Approximate info stats
    # GRAD_UPDATES_PER_AGENT = STEPS_PER_UPDATE * MINIBATCHES *
    # (PPO_RUN_TOT_TIMESTEPS // TOTAL_BATCH_SIZE) * ITER_PER_SELECTION * NUM_PBT_ITER // POPULATION_SIZE

    logging.info("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    # print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "TIMESTAMP_DIR": TIMESTAMP_DIR,
        "RUN_TYPE": RUN_TYPE,
        "EX_DIR": EX_DIR,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "DELAY_MINS": DELAY_MINS,
        "FORCE_KILL": FORCE_KILL,
        "GPU_ID": GPU_ID,
        "PPO_POP_SIZE": PPO_POP_SIZE,
        "HM_POP_SIZE": HM_POP_SIZE,
        "COMBINED_POP_SIZE": COMBINED_POP_SIZE,
        "MDP_PARAMS": {
            "layout_name": LAYOUT_NAME,
            "start_order_list": START_ORDER_LIST,
            "rew_shaping_params": REW_SHAPING_PARAMS
        },  # Note: no caps here is deliberate
        "ENV_PARAMS": {
            "horizon": horizon,
            # "random_start_pos": random_start_pos, # TODO: !!!!
            # "random_start_objs_p": random_start_objs_p
        },
        "FIXED_MDP": FIXED_MDP,
        "PPO_RUN_TOT_TIMESTEPS": PPO_RUN_TOT_TIMESTEPS,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        # "ITER_PER_SELECTION": ITER_PER_SELECTION,
        "RESAMPLE_PROB": RESAMPLE_PROB,
        "MUTATION_FACTORS": MUTATION_FACTORS,
        "PADDED_MDP_SHAPE": PADDED_MDP_SHAPE,
        "MDP_SHAPE_FN": MDP_SHAPE_FN,
        "PROP_EMPTY_FN": PROP_EMPTY_FN,
        "PROP_FEATS_FN": PROP_FEATS_FN,
        "RND_OBJS": RND_OBJS,
        "RND_POS": RND_POS,
        "HYPERPARAMS_TO_MUTATE": HYPERPARAMS_TO_MUTATE,
        # "ORDER_GOAL": ORDER_GOAL, # pk: Why is this not needed?
        "REW_SHAPING_HORIZON": REW_SHAPING_HORIZON,
        # "WEIGHT_HM_HORIZON": WEIGHT_HM_HORIZON,
        "ENTROPY": ENTROPY,
        "GAMMA": GAMMA,
        "sim_threads": sim_threads,
        "TOTAL_BATCH_SIZE": TOTAL_BATCH_SIZE,
        "BATCH_SIZE": BATCH_SIZE,
        "MAX_GRAD_NORM": MAX_GRAD_NORM,
        "LR": LR,
        "VF_COEF": VF_COEF,
        "STEPS_PER_UPDATE": STEPS_PER_UPDATE,
        "MINIBATCHES": MINIBATCHES,
        "CLIPPING": CLIPPING,
        "LAM": LAM,
        "NETWORK_TYPE": NETWORK_TYPE,
        "DISPLAY_TRAINING": DISPLAY_TRAINING,
        "NUM_HIDDEN_LAYERS": NUM_HIDDEN_LAYERS,
        "SIZE_HIDDEN_LAYERS": SIZE_HIDDEN_LAYERS,
        "NUM_FILTERS": NUM_FILTERS,
        "NUM_CONV_LAYERS": NUM_CONV_LAYERS,
        "START_ORIENTATIONS": START_ORIENTATIONS,
        "WAIT_ALLOWED": WAIT_ALLOWED,
        "COUNTER_PICKUP": COUNTER_PICKUP,
        "SAME_MOTION_GOALS": SAME_MOTION_GOALS,
        "BC_AGENT_FOLDER": BC_AGENT_FOLDER,
        "PERSON_PARAMS_HM0": PERSON_PARAMS_HM0,
        "PERSON_PARAMS_HM1": PERSON_PARAMS_HM1,
        # "PERSON_PARAMS_HM2": PERSON_PARAMS_HM2,
        # "PERSON_PARAMS_HM3": PERSON_PARAMS_HM3,
        "REW_SHAPING_PARAMS": REW_SHAPING_PARAMS,
        "MODEL_SAVE_FREQUENCY": MODEL_SAVE_FREQUENCY,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "NUM_EVAL_GAMES": NUM_EVAL_GAMES,
        "TOTAL_STEPS_PER_AGENT": TOTAL_STEPS_PER_AGENT,
        # "grad_updates_per_agent": GRAD_UPDATES_PER_AGENT
        "WEIGHT_HM_MAX": WEIGHT_HM_MAX,
        "INCLUDE_HMS_AFTER": INCLUDE_HMS_AFTER
    }


@ex.named_config
def test_on_pc():

    EX_DIR = "pc_test"

    LAYOUT_NAME = "simple"
    PPO_POP_SIZE = 1  # Try 6?
    HM_POP_SIZE = 1  # Try 2?

    sim_threads = 24
    TOTAL_STEPS_PER_AGENT = 1e3  #
    PPO_RUN_TOT_TIMESTEPS = 100  # pk: How any timesteps each time we "update" the

    TOTAL_BATCH_SIZE = 100  # Local testing reduced
    STEPS_PER_UPDATE = 1
    MINIBATCHES = 1
    NUM_SELECTION_GAMES = 2
    horizon=20

@ex.named_config
def test_on_server1():
    EX_NAME = "test_on_server1"

    SERVER_RUN = True
    LOCAL_TESTING = False
    LAYOUT_NAME = "simple"

    # Try to make things smaller/simpler:
    horizon = 100
    HM_POP_SIZE = 1
    NUM_HIDDEN_LAYERS = 1
    SIZE_HIDDEN_LAYERS = 32
    NUM_FILTERS = 15
    NUM_CONV_LAYERS = 3
    NUM_SELECTION_GAMES = 2


    TOTAL_STEPS_PER_AGENT = 2e5 if not LOCAL_TESTING else 1e3  # 1.5e7

    PPO_RUN_TOT_TIMESTEPS = 1e4 if not LOCAL_TESTING else 100  # 40000
    TOTAL_BATCH_SIZE = 4000 if not LOCAL_TESTING else 100  # 20000

    STEPS_PER_UPDATE = 2 if not LOCAL_TESTING else 1
    MINIBATCHES = 2 if not LOCAL_TESTING else 1

    # FYI: NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * COMBINED_POP_SIZE // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS))

    RND_OBJS = False
    RND_POS = True

@ex.named_config
def test_on_server2():
    EX_NAME = "test_on_server2"

    SERVER_RUN = True

    LAYOUT_NAME = "simple"
    horizon = 100  # 400
    PPO_POP_SIZE = 2  # Try 6?
    HM_POP_SIZE = 1  # Try 2?
    NUM_HIDDEN_LAYERS = 1  # 2-3
    SIZE_HIDDEN_LAYERS = 32  # 64
    NUM_FILTERS = 15  # 25
    NUM_CONV_LAYERS = 3  # 3
    NUM_SELECTION_GAMES = 2  # 10

    TOTAL_STEPS_PER_AGENT = 1e6  # 1.5e7

    PPO_RUN_TOT_TIMESTEPS = 40000  # 40000
    TOTAL_BATCH_SIZE = 20000  # 20000
    STEPS_PER_UPDATE = 8  # 8
    MINIBATCHES = 5  # 5

    # FYI: NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * COMBINED_POP_SIZE // (ITER_PER_SELECTION * PPO_RUN_TOT_TIMESTEPS))

    RND_OBJS = False
    RND_POS = True

@ex.named_config
def test_on_server3():
    """Using NEW DEFAULT params, based on recommendations from Micah
    This run is intended to try and get closer to Micah's results"""

    EX_NAME = "test_on_server3"

    SERVER_RUN = True

    LAYOUT_NAME = "simple"
    PPO_POP_SIZE = 2  # Try 6?
    HM_POP_SIZE = 1  # Try 2?

    NUM_SELECTION_GAMES = 3

@ex.named_config
def test_on_server4():
    """Double the ppo steps, increase reward-shaping horizon, Bigger network"""

    EX_DIR = "first_expts4"

    LAYOUT_NAME = "simple"
    PPO_POP_SIZE = 2  # Try 6?
    HM_POP_SIZE = 1  # Try 2?

    NUM_SELECTION_GAMES = 3

    TOTAL_STEPS_PER_AGENT = 1e7  # This is much larger than Micah said was needed...
    REW_SHAPING_HORIZON = 5e6  # PPO training was still increasing here

    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64

    SEEDS = [8015]  # Seeing whether this makes a difference

@ex.named_config
def first_expts5():
    EX_DIR = "first_expts5"
    EX_NAME = "unnamed"

    LAYOUT_NAME = "simple"
    PPO_POP_SIZE = 2  # Try 6?
    HM_POP_SIZE = 1  # Try 2?

    NUM_SELECTION_GAMES = 3

    TOTAL_STEPS_PER_AGENT = 1e7  # This is much larger than Micah said was needed...
    REW_SHAPING_HORIZON = 3e6  # PPO training was still increasing here
    WEIGHT_HM_HORIZON = 3e6

    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25

    START_ORDER_LIST = ["any"] * 20

    SEEDS = [8015,3554]

@ex.named_config
def first_expts6():
    EX_DIR = "first_expts6"
    #===Remember to name expts: ===#
    EX_NAME = "unnamed"
    #==============================#

    NUM_SELECTION_GAMES = 3
    TOTAL_STEPS_PER_AGENT = 1e7
    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    START_ORDER_LIST = ["any"] * 20
    SEEDS = [8015, 3554]

    LAYOUT_NAME = "simple"
    PPO_POP_SIZE = 2
    HM_POP_SIZE = 2
    REW_SHAPING_HORIZON = 4e6
    WEIGHT_HM_HORIZON = 4e6

@ex.named_config
def second_expts1():
    # Now HMs are only introduced after some steps
    EX_DIR = "second_expts1"
    #===Remember to name expts: ===#
    EX_NAME = "unnamed"
    #==============================#

    # NUM_SELECTION_GAMES = 3
    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    START_ORDER_LIST = ["any"] * 20
    LAYOUT_NAME = "simple"

    # SEEDS = [8015, 3554, 4221, 43, 8640]

    PPO_POP_SIZE = 2
    HM_POP_SIZE = 2
    TOTAL_STEPS_PER_AGENT = 1e7
    REW_SHAPING_HORIZON = 5e6
    INCLUDE_HMS_AFTER = 5e6

    # CHANGES HALF WAY THROUGH:
    NUM_SELECTION_GAMES = 2
    SEEDS = [3554, 4221, 43]
    NUM_EVAL_GAMES = 5

@ex.named_config
def neurips_hps():
    EX_DIR="neurips_hps"
    EX_NAME="unnamed"

    """Note: Only LR and VF_COEFF are different!"""

    PPO_POP_SIZE = 3

    NUM_HIDDEN_LAYERS = 3
    SIZE_HIDDEN_LAYERS = 32
    NUM_FILTERS = 25
    NUM_CONV_LAYERS = 3

    # For simple ("cramped room")
    LAYOUT_NAME = "simple"
    LR = 3e-3  # I was using 1e-3
    VF_COEF = 0.1  # I was using 0.5
    REW_SHAPING_HORIZON = 3e6
    MINIBATCHES = 5
    # minibatch size 4000  # Set by batch size? Which is 20k so this is right?
    PPO_RUN_TOT_TIMESTEPS = 40000
    ENTROPY = 0.5
    GAMMA = 0.99
    LAM = 0.98
    CLIPPING = 0.05
    MAX_GRAD_NORM = 0.1
    STEPS_PER_UPDATE = 8

    INCLUDE_HMS_AFTER = 4.5e6  # 4.5e6 was the TOTAL steps in the paper
    TOTAL_STEPS_PER_AGENT = 1e7

    # Other params I need:
    HM_POP_SIZE = 4
    WEIGHT_HM_MAX = 1
    START_ORDER_LIST = ["any"] * 20
    NUM_SELECTION_GAMES = 3
    SEEDS = [4221, 43, 8640]

@ex.named_config
def third_expts1():
    # Now HMs are only introduced after some steps
    EX_DIR = "third_expts1"
    #===Remember to name expts: ===#
    EX_NAME = "unnamed"
    #==============================#

    # NUM_SELECTION_GAMES = 3
    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    START_ORDER_LIST = ["any"] * 20
    LAYOUT_NAME = "simple"

    # SEEDS = [8015, 3554, 4221, 43, 8640]

    PPO_POP_SIZE = 2
    HM_POP_SIZE = 2
    TOTAL_STEPS_PER_AGENT = 1e7
    REW_SHAPING_HORIZON = 5e6
    INCLUDE_HMS_AFTER = 5e6

    # CHANGES HALF WAY THROUGH:
    NUM_SELECTION_GAMES = 2
    SEEDS = [43, 3554, 4221, 8640]
    NUM_EVAL_GAMES = 5

@ex.named_config
def test_with_bcs():
    EX_DIR = "test_with_bcs"

    START_ORDER_LIST = ["any"] * 20
    LAYOUT_NAME = "simple"

    SEEDS = [43]

    NUM_SELECTION_GAMES = 1
    NUM_EVAL_GAMES = 1

    PPO_POP_SIZE = 2
    HM_POP_SIZE = 2

    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 32
    NUM_FILTERS = 15

    REW_SHAPING_HORIZON = 5e6

    # python pbt/pbt_hms.py with test_with_bcs EX_NAME="test_with_bcs1" TOTAL_STEPS_PER_AGENT=8e6 INCLUDE_HMS_AFTER=4e6
    # python pbt/pbt_hms.py with test_with_bcs EX_NAME="test_with_bcs2" TOTAL_STEPS_PER_AGENT=12e6 INCLUDE_HMS_AFTER=6e6

@ex.named_config
def mem_checks():
    LAYOUT_NAME="simple"
    NUM_HIDDEN_LAYERS=2
    NUM_SELECTION_GAMES=2
    horizon=100
    TOTAL_STEPS_PER_AGENT=1.5e7
    PPO_RUN_TOT_TIMESTEPS=40000
    TOTAL_BATCH_SIZE=20000
    MINIBATCHES=5
    STEPS_PER_UPDATE=8
    SIZE_HIDDEN_LAYERS=64
    NUM_FILTERS=25
    NUM_CONV_LAYERS=3
    TIMESTAMP_DIR=True
    PPO_POP_SIZE=1
    HM_POP_SIZE=1
    INCLUDE_HMS_AFTER=-1
    REW_SHAPING_HORIZON=0
    NUM_EVAL_GAMES=1
    sim_threads=10
    EX_NAME="mem_checks"

@ex.named_config
def fourth_expts1():
    EX_DIR = "fourth_expts1"
    #===Remember to name expts: ===#
    EX_NAME = "unnamed"
    #==============================#

    # Using what I THINK to be the best set of params so far:

    NUM_HIDDEN_LAYERS = 2
    SIZE_HIDDEN_LAYERS = 64
    NUM_FILTERS = 25
    START_ORDER_LIST = ["any"] * 20
    LAYOUT_NAME = "simple"

    PPO_POP_SIZE = 4  # Increasing to e.g. 6 seems to be better
    HM_POP_SIZE = 2
    TOTAL_STEPS_PER_AGENT = 1.5e7  # Might need to go up to 2e7 or even 2.5e7 (but 1.5e7 probs fine for simple)
    REW_SHAPING_HORIZON = 5e6
    INCLUDE_HMS_AFTER = 5e6

    NUM_SELECTION_GAMES = 2
    NUM_EVAL_GAMES = 5

    # IN TWO BATCHES:
    SEEDS = [43, 3554, 4221, 8640]

# Other things to try/play with :
    # Different BC seeds and train/test
    # INCLUDE_HMS_AFTER
    # WEIGHT_HM_MAX > 1
    # Setting LR=5e-4 maybe makes it more stable, but probably needs more time

#=====================================================================================================================#
def pbt_one_run(params, seed):
    # Iterating noptepochs over same batch data but shuffled differently
    # dividing each batch in `nminibatches` and doing a gradient step for each one

    # ---------------- MEM ---------------------#
    mem_snaps = MemSnaps(curr_snap=tm.take_snapshot())
    # ------------------------------------------#

    if not params["TIMESTAMP_DIR"]:
        assert params["EX_NAME"] != "unnamed"  # Otherwise multiple runs will be saved in the same folder and can
        # overwrite each other

    t_start_seed = time.time()

    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")

    #######
    # pbt #
    #######

    mdp = OvercookedGridworld.from_layout_name(**params["MDP_PARAMS"])
    overcooked_env = OvercookedEnv(mdp, **params["ENV_PARAMS"])
    NO_COUNTERS_PARAMS = {
        'start_orientations': params["START_ORIENTATIONS"],
        'wait_allowed': params["WAIT_ALLOWED"],
        'counter_goals': mdp.get_counter_locations(),
        'counter_drop': mdp.get_counter_locations(),
        'counter_pickup': params["COUNTER_PICKUP"],
        'same_motion_goals': params["SAME_MOTION_GOALS"]
    } # This means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    # Print the layouts
    print("Visualise the layouts")
    for _ in range(5):
        overcooked_env.reset()  #pk Qu: Why do we need to reset? Guessed ANS: Because if creating random envs we want
        # to see different random envs rather than just printing the same one 5 times!
        print(overcooked_env)  #pk: We can print this object because the class has function "__repr__"

    # Create gym env:
    gym_env = get_vectorized_gym_env(overcooked_env, 'Overcooked-v0',
                                     featurize_fn=lambda mdp, x: mdp.lossless_state_encoding(x), **params)
    gym_env.update_reward_shaping_param(1.0)  # Start reward shaping from 1
    #TODO pk: Careful here because get_vectorized_gym_env has 'if kwargs["RUN_TYPE"] == "joint_ppo": gym_env.custom_init(base_env,
    # joint_actions=True)'. What does 'joint_ppo' refer to? It's fine because the default in pbt.py is RUN_TYPE = "pbt"?

    # Commented out since switching to Micah's new branch:
    # gym_env.resolve_other_agents = False
    # gym_env.joint_action_model = False
    # gym_env.reward_shaping_factor = 1.0

    # Create annealer for annealing the shaped reward:
    reward_annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])
    # Create annealer for annealing the weighting of hms vs ppo when evaluating the ppo agents
    # weight_annealer = LinearAnnealerZeroToOne(horizon=params["WEIGHT_HM_HORIZON"])

    # POPULATION INITIALIZATION
    combined_pop_size = params["COMBINED_POP_SIZE"]
    ppo_pop_size = params["PPO_POP_SIZE"]

    assert params["TOTAL_STEPS_PER_AGENT"] >= params["INCLUDE_HMS_AFTER"]

    # Make ppo population and also make combined_pbt_pop, which will have both ppo and hm agents:
    ppo_pop = []
    combined_pbt_pop = []
    hm_pop = []
    ppo_agent_names = ['ppo_agent' + str(i) for i in range(ppo_pop_size)]
    for agent_name in ppo_agent_names:
        agent = PPOAgent(agent_name, params, combined_pop_size, gym_env=gym_env)
        ppo_pop.append(agent)
        combined_pbt_pop.append(agent)  # Quicker to make ppo_pop then do combined_pbt_pop = ppo_pop.copy()?

    # Make hm (human model) population, and also add hm agents to combined_pbt_pop
    hm_agent_names = ['hm_agent' + str(i) for i in range(params["HM_POP_SIZE"])]
    hm_number = 0
    for agent_name in hm_agent_names:
        player_index = 99  # We will change player index during training (setting to 99 just in case we forget!)
        agent = HMAgent(params, hm_number, player_index, agent_name)
        combined_pbt_pop.append(agent)
        hm_pop.append(agent)
        hm_number += 1

    # Make the bc agent:
    bc_agent, _ = get_bc_agent_from_saved(params["BC_AGENT_FOLDER"])

    print("Initialized agent models")  #pk: Note: (For now) I'm using 'print' for things I definitely want to print,
    # then using logging.info for information that might or might not be interesting/useful

    # MAIN LOOP
    # @profile  # For memory_profiler
    def pbt_training():
        best_sparse_rew_avg = [-np.Inf]*ppo_pop_size

        for pbt_iter in range(1, params["NUM_PBT_ITER"] + 1):
            print("\n\n\nPBT ITERATION NUM {}".format(pbt_iter))

            # ---------------- MEM ---------------------#
            mem_snaps.prev_snap = mem_snaps.curr_snap
            mem_snaps.curr_snap = None
            mem_snaps.curr_snap = tm.take_snapshot()
            print('\nTop 10 largest files between snapshots:')
            stats = mem_snaps.curr_snap.compare_to(mem_snaps.prev_snap, 'lineno')
            for stat in stats[:10]:
                print(stat)
            print('\nTop 10 largest files overall:')
            stats = mem_snaps.curr_snap.statistics('lineno')
            for stat in stats[:10]:
                print(stat)
            print('\nTraceback for the largest memory between snapshots:')
            stats = mem_snaps.curr_snap.compare_to(mem_snaps.prev_snap, 'traceback')
            top = stats[0]
            print('\n'.join(top.traceback.format()))
            print('\nTraceback for the largest memory overall:')
            stats = mem_snaps.curr_snap.statistics('traceback')
            top = stats[0]
            print('\n'.join(top.traceback.format()))
            print('...')
            # ------------------------------------------#
            # # =========== GARBAGE COLLECT!! ============#
            # gc.collect()
            # # ==========================================#
            # # ---------------- MEM ---------------------#
            # mem_snaps.prev_snap = mem_snaps.curr_snap
            # mem_snaps.curr_snap = None
            # mem_snaps.curr_snap = tm.take_snapshot()
            # print('\nTop 10 largest files between snapshots:')
            # stats = mem_snaps.curr_snap.compare_to(mem_snaps.prev_snap, 'lineno')
            # for stat in stats[:10]:
            #     print(stat)
            # print('\nTop 10 largest files overall:')
            # stats = mem_snaps.curr_snap.statistics('lineno')
            # for stat in stats[:10]:
            #     print(stat)
            # print('\nTraceback for the largest memory between snapshots:')
            # stats = mem_snaps.curr_snap.compare_to(mem_snaps.prev_snap, 'traceback')
            # top = stats[0]
            # print('\n'.join(top.traceback.format()))
            # print('\nTraceback for the largest memory overall:')
            # stats = mem_snaps.curr_snap.statistics('traceback')
            # top = stats[0]
            # print('\n'.join(top.traceback.format()))
            # print('...')
            # # ------------------------------------------#

            # TRAINING PHASE

            # Every ppo agent is paired with every ppo agent, AND after INCLUDE_HMS_AFTER env steps each ppo is also
            # paired with each hm agent (we don't pair hms with hms).
            # We are assuming that all ppo agents have done the same number of env steps (as they should have)
            agent0_env_steps = ppo_pop[0].num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
            print("Agent 0 has done {} steps. We include HMs after {} steps. Total steps per agent: {}".format(
                agent0_env_steps, params["INCLUDE_HMS_AFTER"], params["TOTAL_STEPS_PER_AGENT"]))

            if agent0_env_steps <= params["INCLUDE_HMS_AFTER"]:
                pairs_to_train = list(itertools.product(range(ppo_pop_size), range(ppo_pop_size)))
                weight_param = 0  # Set weight_param = 0 whilst HMs not included, and 1 once they're included
                iter_per_selection = ppo_pop_size * ppo_pop_size
                print("HMs not yet included in training")
            else:
                pairs_to_train = list(itertools.product(range(ppo_pop_size), range(combined_pop_size)))
                weight_param = 1
                iter_per_selection = params["COMBINED_POP_SIZE"] * ppo_pop_size
                print("HMs now included in training")

            for sel_iter in range(pairs_to_train.__len__()):  # For each pair of agents included in pairs_to_train

                # Randomly select agents
                pair_idx = np.random.choice(len(pairs_to_train))
                idx0, idx1 = pairs_to_train.pop(pair_idx)  # This returns a pair & removes that pair from pairs_to_train
                pbt_agent0, pbt_agent1 = ppo_pop[idx0], combined_pbt_pop[idx1]  # NOTE: human model is ALWAYS player 1

                # Training agent 0, leaving agent 1 fixed
                print("Training agent {} (num_ppo_runs: {}) with agent {} fixed or a hm (pbt #{}/{}, sel #{}/{})".format(
                    idx0, pbt_agent0.num_ppo_runs, idx1,
                    pbt_iter, params["NUM_PBT_ITER"], sel_iter+1, iter_per_selection)
                )

                agent_env_steps = pbt_agent0.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                reward_shaping_param = reward_annealer.param_value(agent_env_steps)
                print("Current reward shaping:", reward_shaping_param, "\t Save_dir", params["SAVE_DIR"])
                pbt_agent0.logs["reward_shaping"].append(reward_shaping_param)
                gym_env.update_reward_shaping_param(reward_shaping_param)

                if pbt_agent1.human_model:
                    gym_env.other_agent_hm = True
                else:
                    gym_env.other_agent_hm = False

                gym_env.other_agent = pbt_agent1.get_multi_agent(mlp)  # get_multi_agent=get_agent for PPOAgent

                # Set up display during training
                if params["DISPLAY_TRAINING"] and gym_env.other_agent_hm:
                    random_idx = np.random.randint(params["sim_threads"])
                    gym_env.other_agent[random_idx].display = True

                # Update agent0's model:
                pbt_agent0.update(gym_env)

                # Add the weighting log here, so that it saves:
                # weight_param = weight_annealer.param_value(agent_env_steps+params["PPO_RUN_TOT_TIMESTEPS"])  # The
                # agent has done an extra params["PPO_RUN_TOT_TIMESTEPS"] steps, because .update has been called
                # NEW: weight_param is now set above...
                pbt_agent0.logs["weight_hm"].append(weight_param)

                save_folder = params["SAVE_DIR"] + pbt_agent0.agent_name + '/'
                pbt_agent0.save(save_folder)

                # Observe the agents playing:
                # agent_pair = AgentPair(pbt_agent0.get_agent(mlp), pbt_agent1.get_agent(mlp))
                # overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True,
                #                             display=False, reward_shaping=reward_shaping_param)

            assert len(pairs_to_train) == 0

            # SELECTION PHASE

            print("\nSELECTION PHASE\n")
            # Overwrite worst ppo agent with best ppo agent (mutated), according to a proxy for generalization
            # performance (avg reward across population, with a different, annealed weighting for performance with hms)

            # Dictionary with average returns for each ppo agent when matched with each other agent
            avg_ep_returns_dict = defaultdict(list)
            avg_ep_returns_sparse_dict = defaultdict(list)

            for i in range(ppo_pop_size):
                # Saving each ppo agent model at the end of the pbt iteration
                pbt_agent = ppo_pop[i]
                pbt_agent.update_pbt_iter_logs()

                for j in range(combined_pop_size):
                    # Pairs each ppo agent with all other agents including itself in assessing generalization performance
                    #TODO: The way this is set up we end up simulating each ppo pair twice (e.g. agent 0 is paired with 1
                    # then 1 is paired with 0). This makes the code easier but adds unnecessary extra get_rollouts
                    print("Evaluating agent {} and {}".format(i, j))
                    pbt_agent_other = combined_pbt_pop[j].get_agent(mlp)

                    try:
                        pbt_agent_other.human_model
                        pbt_agent_other.agent_index = 1  # We don't actually need to set this, because AgentPair does it
                        pbt_agent_other.GHM.agent_index = 0  # But we do need to set this!
                    except:
                        AttributeError

                    agent_pair = AgentPair(pbt_agent.get_agent(mlp), pbt_agent_other)
                    trajs = overcooked_env.get_rollouts(agent_pair, params["NUM_SELECTION_GAMES"],
                                                        reward_shaping=reward_shaping_param)
                    dense_rews, sparse_rews, lens = trajs["ep_returns"], trajs["ep_returns_sparse"], trajs["ep_lengths"]
                    rew_per_step = np.sum(dense_rews) / np.sum(lens)

                    if j in range(ppo_pop_size):
                        # If it's paired with a ppo agent:
                        avg_ep_returns_dict[i].append(rew_per_step)
                        avg_ep_returns_sparse_dict[i].append(sparse_rews)
                    elif j in range(ppo_pop_size, combined_pop_size):
                        # Next 3 lines now done above, so that it SAVES
                        # agent_env_steps = pbt_agent.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]  # Same units as rew
                        # weight_param = weight_annealer.param_value(agent_env_steps)
                        # pbt_agent.logs["weight_hm"].append(weight_param)

                        # If it's paired with a hm agent, weight this differently:
                        print("Current HM weighting: {}".format(weight_param))
                        avg_ep_returns_dict[i].append(weight_param*params["WEIGHT_HM_MAX"]*rew_per_step)
                        avg_ep_returns_sparse_dict[i].append(weight_param*params["WEIGHT_HM_MAX"]*sparse_rews)
                    else:
                        raise ValueError('Selecting agent outside the population')

                    # Keep track of performance with each agent
                    log_name = "dense_rew_with_agent{}".format(j)
                    pbt_agent.logs[log_name].append(np.mean(dense_rews))
                    log_name = "sparse_rew_with_agent{}".format(j)
                    pbt_agent.logs[log_name].append(np.mean(sparse_rews))

                if pbt_iter == params["NUM_PBT_ITER"]: # pbt_iter % params["MODEL_SAVE_FREQUENCY"] == 0 or
                    save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/'
                    pbt_agent.save_predictor(save_folder + "pbt_iter{}/".format(pbt_iter))
                    pbt_agent.save(save_folder + "pbt_iter{}/".format(pbt_iter))

            print("AVG ep rewards dict", avg_ep_returns_dict)

            #TODO: This could go in the loop above?
            for i, pbt_agent in enumerate(ppo_pop):
                #TODO: Does this work correctly? Is it needed?:
                pbt_agent.update_avg_rew_per_step_logs(avg_ep_returns_dict[i])

                avg_sparse_rew = np.mean(avg_ep_returns_sparse_dict[i])

                if avg_sparse_rew > best_sparse_rew_avg[i]:
                    best_sparse_rew_avg[i] = avg_sparse_rew
                    agent_name = pbt_agent.agent_name
                    print("New best avg sparse rews {} for agent {}, saving...".format(best_sparse_rew_avg[i],
                                                                                       agent_name))
                    best_save_folder = params["SAVE_DIR"] + agent_name + '/best_sparse/'
                    delete_dir_if_exists(best_save_folder, verbose=True)
                    pbt_agent.save_predictor(best_save_folder)
                    pbt_agent.save(best_save_folder)

            # Get best and worst agents when averaging rew per step across all agents
            best_agent_idx = max(avg_ep_returns_dict, key=lambda key: np.mean(avg_ep_returns_dict[key]))
            worst_agent_idx = min(avg_ep_returns_dict, key=lambda key: np.mean(avg_ep_returns_dict[key]))

            # MUTATION PHASE:
            # Replace worst agent with mutated version of best agent (explore_from does the mutation)

            ppo_pop[worst_agent_idx].explore_from(ppo_pop[best_agent_idx])
            print("Overwrote worst model {} ({} rew) with best model {} ({} rew)"
                    .format(worst_agent_idx, avg_ep_returns_dict[worst_agent_idx],
                            best_agent_idx, avg_ep_returns_dict[best_agent_idx]))

            best_agent = ppo_pop[best_agent_idx].get_agent(mlp)
            best_agent_copy = ppo_pop[best_agent_idx].get_agent(mlp)
            agent_pair = AgentPair(best_agent, best_agent_copy)
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True, display=False,
                                        reward_shaping=reward_shaping_param)

            # EVALUATE BEST AGENT
            # Play best_agent with a bc agent, and record the sparse reward received
            print('#----------------------------#')
            print('best_agent playing with the bc agent...!')
            agent_pair = AgentPair(best_agent, bc_agent)
            trajs = overcooked_env.get_rollouts(agent_pair, num_games=params["NUM_EVAL_GAMES"],
                                        final_state=False, display=False)  # reward shaping not needed
            sparse_rews = trajs["ep_returns_sparse"]
            avg_sparse_rew = np.mean(sparse_rews)
            # Save in agent 0's log:
            ppo_pop[0].logs["best_agent_rew_bc"].append(avg_sparse_rew)
            print('Best agents sparse rew with bc this iter: {}'.format(avg_sparse_rew))
            # To observe play:
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=False)

            # Reward with HM0, for comparison:
            print('best_agent playing with HM0')
            hm0_agent = hm_pop[0].get_agent(mlp)
            hm0_agent.agent_index = 1  # Don't need to set this, as AgentPair does it for us
            hm0_agent.GHM.agent_index = 0  # But we DO need to set this!
            agent_pair = AgentPair(best_agent, hm0_agent)
            trajs = overcooked_env.get_rollouts(agent_pair, num_games=params["NUM_EVAL_GAMES"],
                                                final_state=False, display=False)  # reward shaping not needed
            sparse_rews = trajs["ep_returns_sparse"]
            avg_sparse_rew = np.mean(sparse_rews)
            ppo_pop[0].logs["best_agent_rew_hm0"].append(avg_sparse_rew)
            # To observe play:
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=False)

            # And reward with HM1:
            if params["HM_POP_SIZE"] > 1:
                print('best_agent playing with HM1')
                hm1_agent = hm_pop[1].get_agent(mlp)
                hm1_agent.agent_index = 1  # Don't need to set this, as AgentPair does it for us
                hm1_agent.GHM.agent_index = 0  # But we DO need to set this!
                agent_pair = AgentPair(best_agent, hm1_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=params["NUM_EVAL_GAMES"],
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns_sparse"]
                avg_sparse_rew = np.mean(sparse_rews)
                # Save in agent 0's log:
                ppo_pop[0].logs["best_agent_rew_hm1"].append(avg_sparse_rew)
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=False)
            print('#----------------------------#')

            elapsed_time_this_seed = time.time() - t_start_seed
            #TODO: Currently this is "rough" time because it doesn't factor in time difference between pre and post HM
            # iterations... fix this
            rough_tot_time_this_seed = elapsed_time_this_seed*params["NUM_PBT_ITER"]/pbt_iter
            print('PREDICTED TOTAL TIME FOR THIS SEED: {} sec = {} hrs'.
                  format(np.round(rough_tot_time_this_seed),np.round(rough_tot_time_this_seed/3600, 2)))

    pbt_training()
    reset_tf()
    print(params["SAVE_DIR"])

@ex.automain
def run_pbt(params):
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.WARNING)
    logging.getLogger().setLevel(logging.WARNING)  # pk: Note sure why I need this line too
    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")
    # Pause before starting run
    if params["DELAY_MINS"]:
        delay_before_run(params["DELAY_MINS"])
    for seed in params["SEEDS"]:
        set_global_seed(seed)
        curr_seed_params = params.copy()
        curr_seed_params["SAVE_DIR"] += "seed_{}/".format(seed)
        t_before_run = time.time()
        pbt_one_run(curr_seed_params, seed)
        t_after_run = time.time() - t_before_run
        print('RUN COMPLETE in time {}'.format(t_after_run))

        if params["FORCE_KILL"]:
            print('FORCING KILL...')
            import signal
            import os
            os.kill(os.getpid(), signal.SIGKILL)

        # import sys
        # try:
        #     sys.stderr.close()
        #     sys.exit("Exit")
        # except BrokenPipeError:
        #     sys.stderr.close()
        #     sys.exit("Exit")
        # finally:
        #     sys.stderr.close()
        #     sys.exit("Exit")
        # from sys import exit
        # exit(0)
        # print('Should not print...')

