import time
from argparse import ArgumentParser
from human_aware_rl.ppo.ppo_pop import get_ppo_agent, make_tom_agent
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import numpy as np
import copy
import json

import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

from human_ai_robustness.agent import ToMModel
from human_ai_robustness.import_person_params import import_manual_tom_params
from human_aware_rl.data_dir import DATA_DIR

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

def find_pot_locations(layout):
    pot_locations_dict = {'counter_circuit': [(3, 0), (4, 0)], 'coordination_ring': [(3, 0), (4, 1)], 'room': [(3, 0)],
                            'centre_objects': [(2, 2)], 'centre_pots': [(2, 2), (4, 2)], 'bottleneck': [(4, 4), (5, 4)]}
    return pot_locations_dict[layout]

def get_layout_horizon(layout, horizon_length, test_agent):
    """Return the horizon for given layout/length of task"""
    # extra_time = 0 if test_agent.__class__ is ToMModel else 0  # For test runs, e.g. if we want to give the TOM extra time
    extra_time = 0
    if extra_time != 0:
        print('>>>>>>> Extra time = {} <<<<<<<<'.format(extra_time))
    if horizon_length == 'short':
        return extra_time + 10

    elif horizon_length == 'medium':
        if layout in ['coordination_ring', 'centre_pots']:
            return extra_time + 15
        else:
            return extra_time + 20

    elif horizon_length == 'long':
        if layout == 'counter_circuit':
            return extra_time + 30
        elif layout == 'coordination_ring':
            return extra_time + 25

def make_stationary_tom_agent(mlp):
    """Make a TOM agent that doesn't move: (prob_pausing == 1, prob_random_action=0 (all other params are irrelevant))"""
    compliance, teamwork, retain_goals, wrong_decisions, prob_thinking_not_moving, path_teamwork, \
    rationality_coefficient, prob_pausing, prob_greedy, prob_obs_other, look_ahead_steps = [1] * 11
    tom_agent = ToMModel(mlp=mlp, prob_random_action=0, compliance=compliance, teamwork=teamwork,
                         retain_goals=retain_goals, wrong_decisions=wrong_decisions,
                         prob_thinking_not_moving=prob_thinking_not_moving, path_teamwork=path_teamwork,
                         rationality_coefficient=rationality_coefficient, prob_pausing=prob_pausing,
                         use_OLD_ml_action=False, prob_greedy=prob_greedy, prob_obs_other=prob_obs_other,
                         look_ahead_steps=look_ahead_steps)
    return tom_agent

def make_random_tom_agent(mlp, layout):
    """Make a random TOM agent -- takes random actions"""
    compliance, teamwork, retain_goals, wrong_decisions, prob_thinking_not_moving, path_teamwork, \
    rationality_coefficient, prob_pausing, prob_greedy, prob_obs_other, look_ahead_steps = [99] * 11
    tom_agent = ToMModel(mlp=mlp, prob_random_action=0, compliance=compliance, teamwork=teamwork,
                         retain_goals=retain_goals, wrong_decisions=wrong_decisions,
                         prob_thinking_not_moving=prob_thinking_not_moving, path_teamwork=path_teamwork,
                         rationality_coefficient=rationality_coefficient, prob_pausing=prob_pausing,
                         use_OLD_ml_action=False, prob_greedy=prob_greedy, prob_obs_other=prob_obs_other,
                         look_ahead_steps=look_ahead_steps)
    _, TOM_PARAMS, _ = import_manual_tom_params(layout, 1)
    tom_agent.set_tom_params(None, None, TOM_PARAMS, tom_params_choice=0)
    # Then make it take random steps (set both, just to be sure):
    tom_agent.rationality_coefficient = 0.01
    tom_agent.prob_random_action = 1
    return tom_agent

def make_median_tom_agent(mlp, layout):
    """Make the Median TOM agent -- with params such that is has the median score with other manual param TOMs"""
    compliance, teamwork, retain_goals, wrong_decisions, prob_thinking_not_moving, path_teamwork, \
    rationality_coefficient, prob_pausing, prob_greedy, prob_obs_other, look_ahead_steps = [99] * 11
    tom_agent = ToMModel(mlp=mlp, prob_random_action=0, compliance=compliance, teamwork=teamwork,
                         retain_goals=retain_goals, wrong_decisions=wrong_decisions,
                         prob_thinking_not_moving=prob_thinking_not_moving, path_teamwork=path_teamwork,
                         rationality_coefficient=rationality_coefficient, prob_pausing=prob_pausing,
                         use_OLD_ml_action=False, prob_greedy=prob_greedy, prob_obs_other=prob_obs_other,
                         look_ahead_steps=look_ahead_steps)
    _, TOM_PARAMS, _ = import_manual_tom_params(layout, 1)
    tom_agent.set_tom_params(None, None, TOM_PARAMS, tom_params_choice=0)
    return tom_agent

# def make_cc_standard_test_positions():
#     # Make the standard_test_positions for this layout:
#     standard_test_positions = []
#     # Middle positions:
#     standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (4, 1)})
#     standard_test_positions.append({'r_loc': (4, 1), 'h_loc': (3, 1)})
#     standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (3, 3)})
#     standard_test_positions.append({'r_loc': (3, 3), 'h_loc': (3, 1)})
#     # Side positions:
#     standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (1, 3)})
#     standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (1, 1)})
#     standard_test_positions.append({'r_loc': (6, 1), 'h_loc': (6, 3)})
#     standard_test_positions.append({'r_loc': (6, 3), 'h_loc': (6, 1)})
#     # Diagonal positions:
#     standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (6, 3)})
#     standard_test_positions.append({'r_loc': (6, 3), 'h_loc': (1, 1)})
#     standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (6, 1)})
#     standard_test_positions.append({'r_loc': (6, 1), 'h_loc': (1, 3)})
#     return standard_test_positions
#
# def make_cring_standard_test_positions():
#     # Make the standard_test_positions for CRING:
#     standard_test_positions = []
#     # top-R / bottom-L:
#     standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (3, 2)})
#     standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (2, 1)})
#     standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (1, 1)})
#     standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (3, 3)})
#     # Both near dish/soup:
#     standard_test_positions.append({'r_loc': (1, 2), 'h_loc': (3, 3)})
#     standard_test_positions.append({'r_loc': (2, 3), 'h_loc': (1, 1)})
#     # Diagonal:
#     standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (3, 3)})
#     standard_test_positions.append({'r_loc': (3, 3), 'h_loc': (1, 1)})
#     return standard_test_positions

def make_test_tom_agent(layout, mlp, tom_num):
    """Make a TOM from the VAL OR TRAIN? set used for ppo"""
    VAL_TOM_PARAMS, TRAIN_TOM_PARAMS, _ = import_manual_tom_params(layout, 20)
    tom_agent = make_tom_agent(mlp)
    tom_agent.set_tom_params(None, None, TRAIN_TOM_PARAMS, tom_params_choice=int(tom_num))

    # "OPTIMAL" TOM AGENT SETTINGS:
    # print('>>> Manually overwriting the TOM with an "optimal" TOM <<<')
    # tom_agent.prob_greedy = 1
    # tom_agent.prob_pausing = 0
    # tom_agent.prob_random_action = 0
    # tom_agent.rationality_coefficient = 20
    # tom_agent.path_teamwork = 1
    # tom_agent.prob_obs_other = 0
    # tom_agent.wrong_decisions = 0
    # tom_agent.prob_thinking_not_moving = 0
    # tom_agent.look_ahead_steps = 4
    # tom_agent.retain_goals = 0
    # tom_agent.compliance = 0

    return tom_agent

# def make_default_test_dict():
#     return dict.fromkeys("type", "description", "number", "layouts", "score",
#             "robustness_to_states",  # Whether this test is testing robustness to (potentially unseen) states
#             "robustness_to_agents",  # Whether this test is testing robustness to unseen
#             "memory",  # Is it testing memory
#             "testing_other"
#                          )




def get_r_d_locations_list_1ai(layout):
    """R and Dish locations for test 1ai
    2 R locs near the dish. 2 far away
    Both with one dish and with lots of dishes"""
    if layout == 'counter_circuit':
        return [{'r_loc': (1, 1), 'd_locs': [(0, 1)]},
                {'r_loc': (1, 1), 'd_locs': [(0, 1), (1, 0), (6, 0)]},
                {'r_loc': (6, 1), 'd_locs': [(6, 0)]},
                {'r_loc': (6, 1), 'd_locs': [(0, 1), (1, 0), (6, 0)]}]
    elif layout == 'coordination_ring':
        return [{'r_loc': (2, 1), 'd_locs': [(2, 0)]},
                {'r_loc': (2, 1), 'd_locs': [(2, 0), (1, 0), (0, 1)]},
                {'r_loc': (3, 3), 'd_locs': [(4, 3)]},
                {'r_loc': (3, 3), 'd_locs': [(4, 3), (4, 2), (3, 4)]}]
    elif layout == 'bottleneck':
        return [{'r_loc': (5, 1), 'd_locs': [(6, 1)]},
                {'r_loc': (5, 1), 'd_locs': [(6, 1), (5, 0), (3, 2)]},
                {'r_loc': (1, 1), 'd_locs': [(0, 1)]},
                {'r_loc': (1, 1), 'd_locs': [(0, 1), (1, 0), (0, 2)]}]
    elif layout == 'room':
        return [{'r_loc': (2, 4), 'd_locs': [(0, 4)]},
                {'r_loc': (2, 4), 'd_locs': [(0, 4), (2, 6), (3, 6)]},
                {'r_loc': (4, 1), 'd_locs': [(4, 0)]},
                {'r_loc': (4, 1), 'd_locs': [(4, 0), (5, 0), (6, 2)]}]
    elif layout == 'centre_pots':
        return None
    elif layout == 'centre_objects':
        return None

def run_test_1ai(test_agent, mdp, print_info, stationary_tom_agent, layout, display_runs):
    """1ai) Pick up a dish from a counter: H blocks dispenser (in layouts with only one dispenser)
    Details:    4 different settings for R's location and the location of the dishes
                Both pots cooking
                H holding onion facing South; R holding nothing
                Success: R gets a dish or changes the pot state?

                POSSIBLE ADDITIONS: Give H no object. More positions for R.
    """

    other_player = stationary_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)
    count_success = 0
    num_tests = 0
    subtest_successes = []
    pots = [ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))] # Both cooking
    h_locs_layout = {'counter_circuit': (1, 2), 'coordination_ring': (1, 2), 'bottleneck': (4, 1), 'room': (1, 5)}
    h_loc = h_locs_layout[layout]
    tom_player_state = PlayerState(h_loc, (0, 1), held_object=ObjectState('onion', h_loc))

    r_d_locations_list = get_r_d_locations_list_1ai(layout)

    for i, r_d_locations in enumerate(r_d_locations_list):

        num_tests += 1
        if print_info:
            print('\nR and Dish locations: {}\n'.format(r_d_locations))

        # Arbitrarily but deterministically choose R's orientation:
        ppo_or = orientations[(i+1) % 4]

        # Make the overcooked state:
        ppo_player_state = PlayerState(r_d_locations['r_loc'], ppo_or, held_object=None)

        dish_states = [ObjectState('dish', r_d_locations['d_locs'][k]) for k in range(len(r_d_locations['d_locs']))]
        objects_dict = {pots[k].position: pots[k] for k in range(len(pots))}
        objects_dict.update({dish_states[k].position: dish_states[k] for k in range(len(r_d_locations['d_locs']))})

        state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                    order_list=['any']*100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
        # Check it's a valid state:
        mdp._check_valid_state(state_i)

        env = OvercookedEnv(mdp, start_state_fn=lambda : state_i)
        env.horizon = get_layout_horizon(layout, "medium", test_agent)

        # Play with the tom agent from this state and record score
        agent_pair = AgentPair(test_agent, other_player)
        trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

        # Score in terms of whether the pot state changes:
        state_f = trajs["ep_observations"][0][-1]
        env.state = state_f
        if print_info:
            print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
            print('\nFinal state:\n{}'.format(env))

        if (state_f.players[0].has_object() and state_f.players[0].get_object().name == 'dish') or \
                state_i.all_objects_by_type['soup'] != state_f.all_objects_by_type['soup']:
            if print_info:
                print('PPO has object, or the pot state has changed --> success!')
            count_success += 1
            subtest_successes.append('S')
        else:
            subtest_successes.append('F')

        if print_info:
            print(count_success/num_tests)
            print('Subtest successes: {}'.format(subtest_successes))

    score = count_success/num_tests

    return score





def run_tests(layout, test_agent, tests_to_run, print_info, num_avg, mdp, mlp, display_runs, agent_name):
    """..."""

    # Make TOM test agent:
    if test_agent.__class__ is str and test_agent[:3] == 'tom':
        test_agent = make_test_tom_agent(layout, mlp, tom_num=test_agent[3])
        print('Setting prob_pausing = 0')
        test_agent.prob_pausing = 0

    # Make the TOM agents used for testing:
    stationary_tom_agent = make_stationary_tom_agent(mlp)
    median_tom_agent = make_median_tom_agent(mlp, layout)
    random_tom_agent = make_random_tom_agent(mlp, layout)

    # if layout == 'counter_circuit':
    #     standard_test_positions = make_cc_standard_test_positions()
    # elif layout == 'coordination_ring':
    #     standard_test_positions = make_cring_standard_test_positions()

    results_this_agent = []


    # Test 1ai:
    results_this_agent.append({'type': '1) Interacting with counters',
            'description': 'Pick up a dish from a counter; H blocks dispenser (valid for layouts with one blockable dispenser)',
            'number': '1ai',
            'layouts': ['bottleneck', 'room', 'coordination_ring', 'counter_circuit'],
            'score': None,
            #TODO: None means "not sure if True or False"!
            'robustness_to_states': None,  # Whether this test is testing robustness to (potentially unseen) states
            'robustness_to_agents': None,  # Whether this test is testing robustness to unseen
            'testing_other': ['reacting_to_other_agent', 'off_distribution_game_state']})
    results_this_agent[-1]['score'] = run_test_1ai(...) if layout in results_this_agent[-1]['layouts'] else None  # If this test isn't valid for this layout, then give a score of None




    results_dict_this_agent = {agent_name: results_this_agent}


    # percent_success = [None]*10
    #
    # if "1" in tests_to_run or tests_to_run == "all":
    #     # TEST 1: "H stands still with X, where X CANNOT currently be used"
    #     count_successes = []
    #     for _ in range(num_avg):
    #         count_success, num_tests = h_random_unusable_object(test_agent, mdp, standard_test_positions,
    #                                                             print_info, random_tom_agent, layout, display_runs)
    #         count_successes.append(count_success)
    #     percent_success[1] = round(100 * np.mean(count_successes) / num_tests)
    #     # num_tests_all[1] = num_tests

    # print('RESULT: {}'.format(?))
    return percent_success




# def plot_results(avg_dict, shorten=False):
#
#     y_pos = np.arange(len(avg_dict.keys()))
#     colour = ['B' if i % 2 == 0 else 'R' for i in range(12)]
#     plt.bar(y_pos, avg_dict.values(), align='center', alpha=0.5, color=colour)
#     avg_dict_keys = [list(avg_dict.keys())[i][0:6] for i in range(len(avg_dict))] if shorten else list(avg_dict.keys())
#     plt.xticks(y_pos, avg_dict_keys, rotation=30)
#     plt.ylabel('Avg % success')
#     # plt.title('')
#     plt.show()

def make_average_dict(run_names, results, bests, seeds):
    i = 0
    avg_dict = {}
    for j, run_name in enumerate(run_names):
        for seed in seeds[j]:
            for best in bests:
                b = 'V' if best == 'val' else 'T'
                this_avg = np.mean([results[i][j] for j in range(len(results[i])) if results[i][j] != None])
                avg_dict['{}_{}_{}'.format(run_name, b, seed)] = this_avg
                i += 1
    return avg_dict

# def make_plot_weighted_avg_dict(run_names, results, bests, seeds):
#     i = 0
#     weighted_avg_dict = {}
#     weighting = [0] + [2] * 3 + [1] * 2 + [0] * 2 + [1] * 2  # Give extra weight to tests 1-3 because each has many more sub-tests than the rest, and it would've made sense to split them up
#     for j, run_name in enumerate(run_names):
#         for seed in seeds[j]:
#             for best in bests:
#                 b = 'V' if best == 'val' else 'T'
#                 this_avg = np.sum([results[i][k]*weighting[k] for k in range(len(results[i])) if results[i][k] != None]) \
#                                             / np.sum(weighting)
#                 weighted_avg_dict['{}_{}_{}'.format(run_name, b, seed)] = this_avg
#                 i += 1
#     # plot_results(weighted_avg_dict, shorten=True)
#     return weighted_avg_dict

def make_average_results(results):
    avg_results = []
    for i in range(results):
        this_avg = np.mean([results[i][j] for j in range(len(results[i])) if results[i][j] != None])
        avg_results.append(this_avg)
    return avg_results

def save_results(avg_dict, weighted_avg_dict, results, run_folder, layout):
    timestamp = time.strftime('%Y_%m_%d-%H_%M_%S_')
    filename = DATA_DIR + 'qualitative_expts/{}_avg_dict_{}_{}.txt'.format(run_folder, layout, timestamp)
    with open(filename, 'w') as json_file:
        json.dump(avg_dict, json_file)
    filename = DATA_DIR + 'qualitative_expts/{}_weighted_avg_dict_{}_{}.txt'.format(run_folder, layout, timestamp)
    with open(filename, 'w') as json_file:
        json.dump(weighted_avg_dict, json_file)
    filename = DATA_DIR + 'qualitative_expts/{}_results_{}_{}.txt'.format(run_folder, layout, timestamp)
    with open(filename, 'w') as json_file:
        json.dump(results, json_file)

def make_mdp_mlp(layout):
    # Make the standard mdp for this layout:
    mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=['any'] * 100, cook_time=20,
                                               rew_shaping_params=None)
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)
    return mdp, mlp

def get_bc_agent(seed, layout, mdp, run_on):
    """Return the BC agent for this layout and seed"""
    bc_name = layout + "_bc_train_seed{}".format(seed)
    if run_on == 'local':
        BC_LOCAL_DIR = '/home/pmzpk/bc_runs/'
    bc_agent, _ = get_bc_agent_from_saved(bc_name, unblock_if_stuck=True,
                                           stochastic=True,
                                           overwrite_bc_save_dir=BC_LOCAL_DIR)
    bc_agent.set_mdp(mdp)
    return bc_agent

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected')

def get_run_info(agent_from):
    """Return the seeds and run_names for the run in run_folder"""

    # -------- Choose agents ---------
    if agent_from == 'lstm_expt_cc0':
        run_folder = agent_from
        run_names = ['cc_1tom', 'cc_20tom', 'cc_1bc', 'cc_20bc']
        seeds = [[3264, 4859, 9225]] * 4

    # if agent_from == 'toms':
    #     num_toms = 20
    #     run_names = ['tom{}'.format(i) for i in range(num_toms)]
    #     seeds, bests, shorten, run_folder = [[None]]*num_toms, [None], False, ''
    #
    # elif agent_from == 'bc':
    #     run_names = ['bc']
    #     bests, shorten, run_folder = [None], False, ''
    #     seeds = [[8502, 7786, 9094, 7709]]  # , 103, 5048, 630, 7900, 5309, 8417, 862, 6459, 3459, 1047, 3759, 3806, 8413, 790, 7974, 9845]]  # BCs from ppo_pop

    return run_folder, run_names, seeds


def return_agent_dir(run_on, run_folder):
    """Return the DIR where the agents are saved"""
    if run_on == 'server0':
        return '/home/paul/research/human_ai_robustness/human_ai_robustness/data/ppo_runs/' + run_folder
    elif run_on == 'server1':
        return '/home/paul/agents_to_QT/' + run_folder
    if run_on == 'server_az':
        return '/home/paul/human_ai_robustness/human_ai_robustness/data/ppo_runs/' + run_folder
    elif run_on == 'local':
        return '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/' + run_folder \
            if agent_from != 'toms' else ''

def get_agent_to_test(agent_from, run_name, seed, layout, mdp, run_on):
    """Return the agent that will undergo the qualitative tests"""
    if agent_from == 'toms':
        # The TOM agents are made within run_tests
        return run_name
    elif agent_from == 'bc':
        return get_bc_agent(seed, layout, mdp, run_on)
    else:
        test_agent, config = get_ppo_agent(EXPT_DIR, seed, best='train')
        if config['NETWORK_TYPE'] == 'cnn_lstm_overcooked':
            test_agent.initial_lstm_state = np.zeros([config['sim_threads'], 2 * config['NLSTM']],
                                                     dtype=float)  # The lstm state [has shape num_envs , 2*nlstm] (see baselines_utils.cnn_and_lstm_network)
            test_agent.reset()  # This sets test_agent.state_lstm = test_agent.initial_lstm_state, and sets self.mask=[False, False, ...]
        return test_agent



if __name__ == "__main__":
    """
    Run a qualitative experiment to test robustness of a trained agent. This code works through a suite of tests,
    largely involving putting the test-subject-agent in a specific state, with a specific other player, then seeing if 
    they can still play Overcooked from that position.
    """
    parser = ArgumentParser()
    parser.add_argument("-l", "--layout", help="layout", required=False, default="counter_circuit")
    parser.add_argument("-t", "--tests_to_run", default="all")
    parser.add_argument("-pr", "--print_info", default=False, action='store_true')
    parser.add_argument("-dr", "--display_runs", default=False, action='store_true')
    # parser.add_argument("-pl", "--final_plot")
    parser.add_argument("-a", "--num_avg", type=int, required=False, default=1)
    parser.add_argument("-f", "--agent_from", type=str, required=True, help='e.g. lstm_expt_cc0')
    parser.add_argument("-r", "--run_on", required=False, type=str, help="e.g. server or local", default='local')

    args = parser.parse_args()

    layout, tests_to_run, print_info, display_runs, num_avg, agent_from, run_on = \
        args.layout, args.tests_to_run, str2bool(args.print_info), str2bool(args.display_runs), \
        args.num_avg, args.agent_from, args.run_on

    run_folder, run_names, seeds = get_run_info(agent_from)
    DIR = return_agent_dir(run_on, run_folder)
    mdp, mlp = make_mdp_mlp(layout)

    results = []

    for i, run_name in enumerate(run_names):

        EXPT_DIR = DIR + '/' + run_name + '/'

        for seed in seeds[i]:

            agent_to_test = get_agent_to_test(agent_from, run_name, seed, layout, mdp, run_on)
            agent_name = "{}{}".format(run_name, seed)

            print('\n' + run_name + ' >> seed_' + str(seed))
            time0 = time.perf_counter()
            results.append(run_tests(layout, agent_to_test, tests_to_run, print_info, num_avg, mdp, mlp, display_runs, agent_name))
            print('Time for this agent: {}'.format(time.perf_counter() - time0))

    """POST PROCESSING..."""
    # avg_dict = make_average_dict(run_names, results, bests, seeds)
    # if final_plot is True:
    #     plot_results(avg_dict, shorten)
    # weighted_avg_dic = make_plot_weighted_avg_dict(run_names, results, bests, seeds)
    # # save_results(avg_dict, weighted_avg_dic, results, run_folder, layout)
    # print('\nFinal average dict: {}'.format(avg_dict))
    # print('\nFinal wegihted avg: {}'.format(weighted_avg_dic))
    print('\nFinal "results": {}'.format(results))
