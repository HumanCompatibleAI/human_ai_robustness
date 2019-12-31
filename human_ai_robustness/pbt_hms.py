import os, time, logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress a tensorflow error
import numpy as np
import tensorflow as tf
from collections import defaultdict
from tensorflow.saved_model import simple_save

from sacred import Experiment
from sacred.observers import FileStorageObserver
from human_aware_rl.data_dir import DATA_DIR

# For sacred experiments -- NEEDS TO GO HERE, before importing other modules, otherwise sacred gives an error:
PBT_DATA_DIR = DATA_DIR + "pbt_hms_runs/"
ex = Experiment('PBT_HMS')
ex.observers.append(FileStorageObserver.create(PBT_DATA_DIR))

from overcooked_ai_py.utils import save_dict_to_file, load_dict_from_file
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair
from human_ai_robustness.agent import ToMModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from human_aware_rl.utils import create_dir_if_not_exists, delete_dir_if_exists, \
    reset_tf, set_global_seed, find_dense_reward_fn
from human_aware_rl.baselines_utils import create_model, get_vectorized_gym_env, \
    update_model, get_agent_from_model, save_baselines_model, overwrite_model, \
    load_baselines_model, LinearAnnealer, delay_before_run
from human_ai_robustness.human_ai_robustness_utils import LinearAnnealerZeroToOne
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_ai_robustness.import_person_params import import_person_params
from human_aware_rl.utils import convert_layout_names_if_required

# Suppress warnings:
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # This surpresses tf errors, mainly this one was getting in the way: (E
# tensorflow/core/common_runtime/bfc_allocator.cc:373] tried to deallocate nullptr)
os.environ['KMP_WARNINGS'] = 'off'  # This is meant to suppress the "OMP: Info #250: KMP_AFFINITY" error
# import tensorflow.python.util.deprecation as deprecation
# deprecation._PRINT_DEPRECATION_WARNINGS = False


class ToMAgent(object):
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
        #TODO: This can be made much briefer (something like [get_agent(mlp) for i in range(...)] ?!
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
            "prob_play_HM": []
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
            self.logs["best_agent_rew_bc2"] = []
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

    #TODO: Should change get_agent to have a variable number of parameters (we need mlp above but not here)
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

@ex.config
def my_config():
    LOCAL_TESTING = False

    GPU_ID = 0
    RUN_TYPE = "pbt"  #pk: Needed for making gym envs

    # Saving
    EX_DIR = "unnamed" if not LOCAL_TESTING else "test"
    EX_NAME = "unnamed"
    TIMESTAMP_DIR = False if not LOCAL_TESTING else True
    if TIMESTAMP_DIR:
        SAVE_DIR = PBT_DATA_DIR + EX_DIR + "/" + time.strftime('%d-%m_%H:%M:%S') + "/"
    else:
        SAVE_DIR = PBT_DATA_DIR + EX_DIR + "/" + EX_NAME + "/"
    logging.info("Saving data to ", SAVE_DIR)
    MODEL_SAVE_FREQUENCY = 10

    DELAY_MINS = None
    QUIT_WO_MULTIKILL = False
    MULTIKILL_WO_QUIT = True

    # PBT params
    sim_threads = 50 if not LOCAL_TESTING else 4
    PPO_POP_SIZE = 2 if not LOCAL_TESTING else 1
    HM_POP_SIZE = 2 if not LOCAL_TESTING else 1
    COMBINED_POP_SIZE = PPO_POP_SIZE + HM_POP_SIZE
    # ITER_PER_SELECTION = COMBINED_POP_SIZE * PPO_POP_SIZE  # How many pairings and model training updates before the
    # worst model is overwritten.
    PPO_RUN_TOT_TIMESTEPS = 50000 if not LOCAL_TESTING else 100  #pk: How any timesteps each time we "update" the
    # model using .update function. Changed from 40k to 50k

    #---------------#
    # Work out how many PBT ITERs:
    UPDATES_EACH_ITER = 10 if not LOCAL_TESTING else 1  # Per iter, how many times to pair up an agent then do an update
    TOTAL_STEPS_PER_AGENT = 2e7 if not LOCAL_TESTING else 1e3  # Default changed to 2e7 instead of 5e6. Total steps
    # should be a multiple of UPDATES_EACH_ITER * PPO_RUN_TOT_TIMESTEPS
    NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT / (UPDATES_EACH_ITER * PPO_RUN_TOT_TIMESTEPS))
    #---------------#
    PROB_PLAY_HM_HORIZON = TOTAL_STEPS_PER_AGENT  # In units of steps

    TOTAL_BATCH_SIZE = 20000 if not LOCAL_TESTING else 100
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

    # Whether same seed is used in all parallel enviornment
    REPRODUCIBLE = False

    DISPLAY_TRAINING = False  # Display the overcooke game (for 1 random SIM_THREAD of the HM) during training

    # # Weight the hm agents differently when assessing performance of agents. This gives the maximum weighting given to
    # #the human models, compared to the ppos, during selection in pbt. We anneal the weight from 0 to this max value.
    # WEIGHT_HM_MAX = 1

    # Mdp params
    FIXED_MDP = True
    LAYOUT_NAME = "simple"  # BC automatically changes when layout changes
    horizon = 400 if not LOCAL_TESTING else 20
    START_ORDER_LIST = ["any"] * 20  # Changed from 10
    RND_OBJS = 0.0
    RND_POS = False
    REW_SHAPING_HORIZON = 5e6
    # WEIGHT_HM_HORIZON = 5e6  # Around 5e6 env steps the PPO SEEMS to not get much better (??????)

    if not LOCAL_TESTING:
        assert horizon == TOTAL_BATCH_SIZE / sim_threads  # BATCH_SIZE = TOTAL_BATCH_SIZE / sim_threads we want
        # BATCH_SIZE == horizon, so that exactly one episode is done each update

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

    NUM_HIDDEN_LAYERS = 2 if not LOCAL_TESTING else 1
    SIZE_HIDDEN_LAYERS = 64 if not LOCAL_TESTING else 32
    NUM_FILTERS = 25 if not LOCAL_TESTING else 15
    NUM_CONV_LAYERS = 3 if not LOCAL_TESTING else 1

    SEEDS = [0]
    NUM_SELECTION_GAMES = 2 if not LOCAL_TESTING else 1
    NUM_EVAL_GAMES = 5 if not LOCAL_TESTING else 1 # Number of games used when evaluating the best_agent in each iter

    # NO_COUNTER_PARAMS:
    START_ORIENTATIONS = False
    WAIT_ALLOWED = False
    COUNTER_PICKUP = []
    SAME_MOTION_GOALS = True

    #--------------------------------------------#
    # Micah uses different seeds for different layouts: "simple": [2, 0], "unident_s": [1, 1],
    # "random1": [0, 1], "random0": [4, 1], "random3": [1, 3] <-- given in format [train, test]
    if LAYOUT_NAME == 'simple':
        bc_seed = '2'
        bc_seed2 = '0'
    elif LAYOUT_NAME == 'unident_s':
        bc_seed = '1'
        bc_seed2 = '1'
    elif LAYOUT_NAME == 'random1':
        bc_seed = '0'
        bc_seed2 = '1'
    elif LAYOUT_NAME == 'random0':
        bc_seed = '4'
        bc_seed2 = '1'
    elif LAYOUT_NAME == 'random3':
        bc_seed = '1'
        bc_seed2 = '3'
    else:
        print('Layout not included in BC dataset')
    BC_AGENT_FOLDER = '{}_bc_{}_seed{}'.format(LAYOUT_NAME, 'train', bc_seed)  # BC model to test with
    BC_AGENT_FOLDER2 = '{}_bc_{}_seed{}'.format(LAYOUT_NAME, 'test', bc_seed2)  # BC model to test with

    # Personality parameters for the human models:
    PERSON_PARAMS_HM0, PERSON_PARAMS_HM1, PERSON_PARAMS_HM2, PERSON_PARAMS_HM3 \
        = import_person_params(LAYOUT_NAME)

    BATCH_SIZE = TOTAL_BATCH_SIZE // sim_threads

    logging.info("Total steps per agent", TOTAL_STEPS_PER_AGENT)
    # print("Grad updates per agent", GRAD_UPDATES_PER_AGENT)

    OTHER_AGENT_TYPE = None  # Needed for compatability with ppo2.py

    params = {
        "LOCAL_TESTING": LOCAL_TESTING,
        "TIMESTAMP_DIR": TIMESTAMP_DIR,
        "RUN_TYPE": RUN_TYPE,
        "EX_DIR": EX_DIR,
        "EX_NAME": EX_NAME,
        "SAVE_DIR": SAVE_DIR,
        "DELAY_MINS": DELAY_MINS,
        "QUIT_WO_MULTIKILL": QUIT_WO_MULTIKILL,
        "MULTIKILL_WO_QUIT": MULTIKILL_WO_QUIT,
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
        "UPDATES_EACH_ITER": UPDATES_EACH_ITER,
        "NUM_PBT_ITER": NUM_PBT_ITER,
        "PROB_PLAY_HM_HORIZON": PROB_PLAY_HM_HORIZON,
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
        "REPRODUCIBLE": REPRODUCIBLE,
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
        "BC_AGENT_FOLDER2": BC_AGENT_FOLDER2,
        "PERSON_PARAMS_HM0": PERSON_PARAMS_HM0,
        "PERSON_PARAMS_HM1": PERSON_PARAMS_HM1,
        "PERSON_PARAMS_HM2": PERSON_PARAMS_HM2,
        "PERSON_PARAMS_HM3": PERSON_PARAMS_HM3,
        "REW_SHAPING_PARAMS": REW_SHAPING_PARAMS,
        "MODEL_SAVE_FREQUENCY": MODEL_SAVE_FREQUENCY,
        "SEEDS": SEEDS,
        "NUM_SELECTION_GAMES": NUM_SELECTION_GAMES,
        "NUM_EVAL_GAMES": NUM_EVAL_GAMES,
        "TOTAL_STEPS_PER_AGENT": TOTAL_STEPS_PER_AGENT,
        # "WEIGHT_HM_MAX": WEIGHT_HM_MAX
        "OTHER_AGENT_TYPE": OTHER_AGENT_TYPE,
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

    # FYI *OLD*: NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * COMBINED_POP_SIZE // (ITER_PER_SELECTION *
    # PPO_RUN_TOT_TIMESTEPS))

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

    # FYI *OLD*: NUM_PBT_ITER = int(TOTAL_STEPS_PER_AGENT * COMBINED_POP_SIZE // (ITER_PER_SELECTION *
    # PPO_RUN_TOT_TIMESTEPS))

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

    params["CURR_SEED"] = seed

    if not params["TIMESTAMP_DIR"]:
        print('WARNING! Make sure the chosen directory has not been used previously:')
        print(params["SAVE_DIR"])
        assert params["EX_NAME"] != "unnamed"  # Otherwise multiple runs will be saved in the same folder and can
        # overwrite each other

    t_start_seed = time.time()

    create_dir_if_not_exists(params["SAVE_DIR"])
    save_dict_to_file(params, params["SAVE_DIR"] + "config")

    params["MDP_PARAMS"]["layout_name"] = convert_layout_names_if_required(params["MDP_PARAMS"]["layout_name"])
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
    gym_env.run_type = params["RUN_TYPE"]

    #######
    # pbt #
    #######

    # Create annealer for annealing the shaped reward:
    reward_annealer = LinearAnnealer(horizon=params["REW_SHAPING_HORIZON"])
    # # Create annealer for annealing the weighting of hms vs ppo when evaluating the ppo agents
    # weight_annealer = LinearAnnealerZeroToOne(horizon=params["WEIGHT_HM_HORIZON"])

    prob_play_HM_annealer = LinearAnnealerZeroToOne(horizon=params["PROB_PLAY_HM_HORIZON"])

    # POPULATION INITIALIZATION

    combined_pop_size = params["COMBINED_POP_SIZE"]
    ppo_pop_size = params["PPO_POP_SIZE"]
    hm_pop_size = params["HM_POP_SIZE"]

    # Make ppo population and also make combined_pbt_pop, which will have both ppo and hm agents:
    ppo_pop = []
    # combined_pbt_pop = []  # No longer needed!
    hm_pop = []
    ppo_agent_names = ['ppo_agent' + str(i) for i in range(ppo_pop_size)]
    for agent_name in ppo_agent_names:
        agent = PPOAgent(agent_name, params, combined_pop_size, gym_env=gym_env)
        ppo_pop.append(agent)
        # combined_pbt_pop.append(agent)  # Quicker to make ppo_pop then do combined_pbt_pop = ppo_pop.copy()?
    # Make hm (human model) population, and also add hm agents to combined_pbt_pop:
    hm_agent_names = ['hm_agent' + str(i) for i in range(params["HM_POP_SIZE"])]
    hm_number = 0
    for agent_name in hm_agent_names:
        player_index = 99  # We will change player index during training (setting to 99 just in case we forget!)
        agent = ToMAgent(params, hm_number, player_index, agent_name)
        # combined_pbt_pop.append(agent)
        hm_pop.append(agent)
        hm_number += 1

    # Make the bc agent:
    bc_agent, _ = get_bc_agent_from_saved(params["BC_AGENT_FOLDER"], unblock_if_stuck=True)
    bc_agent2, _ = get_bc_agent_from_saved(params["BC_AGENT_FOLDER2"], unblock_if_stuck=True)

    print("Initialized agent models")  #pk: Note: (For now) I'm using 'print' for things I definitely want to print,
    # then using logging.info for information that might or might not be interesting/useful

    # MAIN LOOP

    def pbt_training():
        best_sparse_rew_avg = [-np.Inf]*ppo_pop_size

        for pbt_iter in range(1, params["NUM_PBT_ITER"]+1):
            print("\n\n\nPBT ITERATION NUM {}".format(pbt_iter))

            # TRAINING PHASE

            # For each ppo agent in the pop:
            for idx0, pbt_agent0 in enumerate(ppo_pop):

                # Loop over the number of updates each ppo agent will do per iter
                for update_num in range(1, params["UPDATES_EACH_ITER"]+1):

                    # Number of steps done determines the prob of playing with a HM during training & the rew shaping:
                    # (Put this here so that the logs can be plotted alongside the logs from the ppo
                    agent_env_steps = pbt_agent0.num_ppo_runs * params["PPO_RUN_TOT_TIMESTEPS"]
                    prob_play_HM = prob_play_HM_annealer.param_value(agent_env_steps)
                    pbt_agent0.logs["prob_play_HM"].append(prob_play_HM)
                    print("Probability of this ppo agent playing with a HM during training:", prob_play_HM)
                    reward_shaping_param = reward_annealer.param_value(agent_env_steps)
                    print("Current reward shaping:", reward_shaping_param)
                    pbt_agent0.logs["reward_shaping"].append(reward_shaping_param)
                    gym_env.update_reward_shaping_param(reward_shaping_param)

                    # With probability = 1-prob_play_HM, train with a (random) PPO. Otherwise train with a HM:
                    #TODO: The 4x random numbers below make it non-reproducible. Make random number vector at start?
                    # Or do proportion instead of probability?
                    if np.random.random() > prob_play_HM:
                        idx1 = np.random.randint(0, ppo_pop_size)
                        pbt_agent1 = ppo_pop[idx1]
                        #TODO: Make the next few lines into a function:
                        update_number = idx0 * params["UPDATES_EACH_ITER"] + update_num
                        total_updates_each_iter = params["UPDATES_EACH_ITER"] * ppo_pop_size
                        print("Training agent {} (num_ppo_runs: {}) with PPO agent {} fixed (pbt #{}/{}, upd #{}/{})".
                            format(idx0, pbt_agent0.num_ppo_runs, idx1, pbt_iter, params["NUM_PBT_ITER"],
                                   update_number, total_updates_each_iter))
                    else:
                        idx1 = np.random.randint(0, hm_pop_size)
                        pbt_agent1 = hm_pop[idx1]
                        update_number = idx0 * params["UPDATES_EACH_ITER"] + update_num
                        total_updates_each_iter = params["UPDATES_EACH_ITER"] * ppo_pop_size
                        print("Training agent {} (num_ppo_runs: {}) with HM agent {} fixed (pbt #{}/{}, upd #{}/{})".
                              format(idx0, pbt_agent0.num_ppo_runs, idx1, pbt_iter, params["NUM_PBT_ITER"],
                                     update_number, total_updates_each_iter))

                    # Now have pbt_agent0 and pbt_agent1 ready for training...

                    if pbt_agent1.human_model:
                        gym_env.other_agent_tom = True
                    else:
                        gym_env.other_agent_tom = False

                    gym_env.other_agent = pbt_agent1.get_multi_agent(mlp)  # get_multi_agent=get_agent for PPOAgent

                    # Set up display during training
                    if params["DISPLAY_TRAINING"] and gym_env.other_agent_tom:
                        random_idx = np.random.randint(params["sim_threads"])
                        gym_env.other_agent[random_idx].display = True

                    # Update agent0's model:
                    pbt_agent0.update(gym_env)

                    save_folder = params["SAVE_DIR"] + pbt_agent0.agent_name + '/'
                    pbt_agent0.save(save_folder)

            # SELECTION PHASE

            print("\nSELECTION PHASE\n")
            # Overwrite worst ppo agent with best ppo agent (mutated), according to a proxy for generalization
            # performance (avg reward across (subsets of the) population)

            # Dictionary with average returns for each ppo agent when matched with other agents
            avg_ep_returns_dict = defaultdict(list)
            avg_ep_returns_sparse_dict = defaultdict(list)

            for i, pbt_agent in enumerate(ppo_pop):
                # Saving each ppo agent model at the end of the pbt iteration
                pbt_agent.update_pbt_iter_logs()

                # When the prob of training with HMs is < 0.5, then do selection only with PPOs. Otherwise, do selection
                # only with HMs. I.e. we only use EITHER the ppo pop OR the hm pop during selection. This speeds up
                # the runs. And later (earlier) we only really care about performance with HMs (PPOs) anyway.

                if prob_play_HM < 0.5:  # prob_play_HM will be set by only the last ppo agent's latest update. As all
                    # ppo agents have the same number of updates, then prob_play_HM should be the same for all
                    # agents. Note that prob_play_HM will be larger than when the PPO was randomly choosing to play
                    # with HMs. So we use HMs for selection v slightly earlier than when using them for updating models.

                    # Play only with PPO agents:
                    for j in range(ppo_pop_size):

                        print("Evaluating agent {} with PPO agent {}".format(i, j))
                        pbt_agent_other = ppo_pop[j].get_agent(mlp)
                        agent_pair = AgentPair(pbt_agent.get_agent(mlp), pbt_agent_other)
                        trajs = overcooked_env.get_rollouts(agent_pair, params["NUM_SELECTION_GAMES"],
                                                            reward_shaping=reward_shaping_param,
                                                            metadata_fn=find_dense_reward_fn(reward_shaping_param))
                        dense_rews, sparse_rews, lens = trajs["metadatas"]["ep_returns_shaped"], trajs["ep_returns"], \
                                                        trajs["ep_lengths"]
                        rew_per_step = np.sum(dense_rews) / np.sum(lens)
                        avg_ep_returns_dict[i].append(rew_per_step)
                        avg_ep_returns_sparse_dict[i].append(sparse_rews)

                elif prob_play_HM >= 0.5:

                    # Play only with HM agents:
                    for j in range(hm_pop_size):

                        print("Evaluating agent {} with HM agent {}".format(i, j))
                        pbt_agent_other = hm_pop[j].get_agent(mlp)
                        pbt_agent_other.agent_index = 1  # We don't actually need to set this, because AgentPair does it
                        pbt_agent_other.GHM.agent_index = 0  # But we do need to set this!
                        agent_pair = AgentPair(pbt_agent.get_agent(mlp), pbt_agent_other)
                        trajs = overcooked_env.get_rollouts(agent_pair, params["NUM_SELECTION_GAMES"],
                                                            reward_shaping=reward_shaping_param,
                                                            metadata_fn=find_dense_reward_fn(reward_shaping_param))
                        dense_rews, sparse_rews, lens = trajs["metadatas"]["ep_returns_shaped"], trajs["ep_returns"], \
                                                        trajs["ep_lengths"]
                        rew_per_step = np.sum(dense_rews) / np.sum(lens)
                        avg_ep_returns_dict[i].append(rew_per_step)
                        avg_ep_returns_sparse_dict[i].append(sparse_rews)

                #TODO: Keep track of performance with each agent? Note: we'd only have info about the ppo agents for
                # the first half and only the hm agents for the second half (set rew=0 otherwise?)

                # log_name = "dense_rew_with_agent{}".format(j)
                # pbt_agent.logs[log_name].append(np.mean(dense_rews))
                # log_name = "sparse_rew_with_agent{}".format(j)
                # pbt_agent.logs[log_name].append(np.mean(sparse_rews))

                # if pbt_iter == params["NUM_PBT_ITER"]: # pbt_iter % params["MODEL_SAVE_FREQUENCY"] == 0 or
                #     save_folder = params["SAVE_DIR"] + pbt_agent.agent_name + '/'
                #     pbt_agent.save_predictor(save_folder + "pbt_iter{}/".format(pbt_iter))
                #     pbt_agent.save(save_folder + "pbt_iter{}/".format(pbt_iter))

            print("AVG ep rewards dict:", avg_ep_returns_dict)

            #TODO: This could go in the loop above?
            #TODO: PK: I commented most of this out because I'm not using it

            for i, pbt_agent in enumerate(ppo_pop):
            #     #TODO: Does this work correctly? Is it needed?:
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

            print('\nbest_agent playing with itself...!')
            best_agent = ppo_pop[best_agent_idx].get_agent(mlp)
            best_agent_copy = ppo_pop[best_agent_idx].get_agent(mlp)
            agent_pair = AgentPair(best_agent, best_agent_copy)
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=True, display=True,
                                        reward_shaping=reward_shaping_param)

            # EVALUATE BEST AGENT
            print('\n#----------------------------#')
            if ((pbt_iter + 1) >= params["NUM_PBT_ITER"]) and not params["LOCAL_TESTING"]:
                print('FINAL TWO ITERs! So we do 100 eval games')
                num_eval_games = 100
            else:
                num_eval_games = params["NUM_EVAL_GAMES"]

            # TODO: Make "play with agent X" into a helper function (this bit is absurbly long, now that the index
            #  depends on the iteration being odd/even!)
            # On even iterations, PPO is player 0:
            if (pbt_iter % 2) == 0:

                # Reward with HM0, for comparison with BC (which plays below):
                print('best_agent playing with HM0...')
                print('PPO is PLAYER 0')
                hm0_agent = hm_pop[0].get_agent(mlp)
                hm0_agent.agent_index = 1  # Don't need to set this, as AgentPair does it for us
                hm0_agent.GHM.agent_index = 0  # But we DO need to set this!
                agent_pair = AgentPair(best_agent, hm0_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                ppo_pop[0].logs["best_agent_rew_hm0"].append(avg_sparse_rew)
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

                # And reward with HM1:
                #TODO: Could make it play with every HM, but only for 1 game (and display some of them)?
                if params["HM_POP_SIZE"] > 1:
                    print('\n\nbest_agent playing with HM1 (only 1 evaluation game with HM1)')
                    print('PPO is PLAYER 0')
                    hm1_agent = hm_pop[1].get_agent(mlp)
                    hm1_agent.agent_index = 1  # Don't need to set this, as AgentPair does it for us
                    hm1_agent.GHM.agent_index = 0  # But we DO need to set this!
                    agent_pair = AgentPair(best_agent, hm1_agent)
                    trajs = overcooked_env.get_rollouts(agent_pair, num_games=1,
                                                        final_state=False, display=True)  # reward shaping not needed
                    sparse_rews = trajs["ep_returns"]
                    avg_sparse_rew = np.mean(sparse_rews)
                    # Save in agent 0's log:
                    ppo_pop[0].logs["best_agent_rew_hm1"].append(avg_sparse_rew)
                    # To observe play:
                    # overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=False)

                # Play best_agent with a bc agent, and record the sparse reward received
                # (Putting BC last so that it's easier to find and observe the agent playing with BC

                print('\n\nbest_agent playing with BC agent (train, seed 2)...')
                print('PPO is PLAYER 0')
                agent_pair = AgentPair(best_agent, bc_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                # Save in agent 0's log:
                ppo_pop[0].logs["best_agent_rew_bc"].append(avg_sparse_rew)
                print('Best agents sparse rew with BC this iter: {}'.format(avg_sparse_rew))
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

                # Play with a different BC agent:
                print('\n\nbest_agent playing with BC agent 2 (test, seed 0)...')
                print('PPO is PLAYER 0')
                agent_pair = AgentPair(best_agent, bc_agent2)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                # Save in agent 0's log:
                ppo_pop[0].logs["best_agent_rew_bc2"].append(avg_sparse_rew)
                print('Best agents sparse rew with BC2 this iter: {}'.format(avg_sparse_rew))
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

            # On odd iterations, PPO is player 1:
            elif (pbt_iter % 2) == 1:

                # Reward with HM0, for comparison with BC (which plays below):
                print('best_agent playing with HM0...')
                print('PPO is PLAYER 1')
                hm0_agent = hm_pop[0].get_agent(mlp)
                hm0_agent.agent_index = 0  # Don't need to set this, as AgentPair does it for us
                hm0_agent.GHM.agent_index = 1  # But we DO need to set this!
                agent_pair = AgentPair(hm0_agent, best_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                ppo_pop[0].logs["best_agent_rew_hm0"].append(avg_sparse_rew)
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

                # And reward with HM1:
                # TODO: Could make it play with every HM, but only for 1 game (and display some of them)?
                if params["HM_POP_SIZE"] > 1:
                    print('\n\nbest_agent playing with HM1 (only 1 evaluation game with HM1)')
                    print('PPO is PLAYER 1')
                    hm1_agent = hm_pop[1].get_agent(mlp)
                    hm1_agent.agent_index = 0  # Don't need to set this, as AgentPair does it for us
                    hm1_agent.GHM.agent_index = 1  # But we DO need to set this!
                    agent_pair = AgentPair(hm1_agent, best_agent)
                    trajs = overcooked_env.get_rollouts(agent_pair, num_games=1,
                                                        final_state=False,
                                                        display=True)  # reward shaping not needed
                    sparse_rews = trajs["ep_returns"]
                    avg_sparse_rew = np.mean(sparse_rews)
                    # Save in agent 0's log:
                    ppo_pop[0].logs["best_agent_rew_hm1"].append(avg_sparse_rew)
                    # To observe play:
                    # overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=False)

                # Play best_agent with a bc agent, and record the sparse reward received
                # (Putting BC last so that it's easier to find and observe the agent playing with BC

                print('\n\nbest_agent playing with BC agent (train, seed 2)...')
                print('PPO is PLAYER 1')
                agent_pair = AgentPair(bc_agent, best_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                # Save in agent 0's log:
                ppo_pop[0].logs["best_agent_rew_bc"].append(avg_sparse_rew)
                print('Best agents sparse rew with BC this iter: {}'.format(avg_sparse_rew))
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

                # Play with a different BC agent:
                print('\n\nbest_agent playing with BC agent 2 (test, seed 0)...')
                print('PPO is PLAYER 1')
                agent_pair = AgentPair(bc_agent2, best_agent)
                trajs = overcooked_env.get_rollouts(agent_pair, num_games=num_eval_games,
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)
                # Save in agent 0's log:
                ppo_pop[0].logs["best_agent_rew_bc2"].append(avg_sparse_rew)
                print('Best agents sparse rew with BC2 this iter: {}'.format(avg_sparse_rew))
                # To observe play:
                overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)
            print('#----------------------------#\n')

            elapsed_time_this_seed = time.time() - t_start_seed
            pred_tot_time_this_seed = elapsed_time_this_seed*params["NUM_PBT_ITER"]/pbt_iter
            print('PREDICTED TOTAL TIME FOR THIS SEED: {} sec = {} hrs'.
                  format(np.round(pred_tot_time_this_seed), np.round(pred_tot_time_this_seed/3600, 2)))

    pbt_training()
    reset_tf()
    print(params["SAVE_DIR"])

@ex.automain
def run_pbt(params):

    import sys
    sys.exit('\n----> Before using, need to fully debug & check pbt_hms because there have been several changes made '
             'to baselines (ppo2 & runner?) when making ppo work for PPO_{TOMs} <----')

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
        print('SEED COMPLETE in time {}'.format(t_after_run))

    if params["MULTIKILL_WO_QUIT"]:
        print('Closing the multiproc children (but not quitting properly)...')
        # Kill children:
        import psutil
        import multiprocessing
        def kill_children_parent():
            for proc in multiprocessing.active_children():
                proc.terminate()
            parent = psutil.Process(os.getpid())
            for child in parent.children(recursive=True):
                child.kill()
            parent.kill()
        kill_children_parent
    if params["QUIT_WO_MULTIKILL"]:
        print('Quitting (kill) but not closing the multiproc children properly...')
        import signal
        import os
        os.kill(os.getpid(), signal.SIGKILL)

