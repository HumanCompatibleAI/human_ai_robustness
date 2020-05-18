import random
import time

from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import copy
import numpy as np
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from human_aware_rl.ppo.ppo_pop import make_tom_agent
from human_ai_robustness.import_person_params import import_manual_tom_params
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from human_aware_rl.human.process_dataframes import get_human_human_trajectories

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

def make_tom_pop(prob_pausing_factor):
    """Make a population of TOMs
    params: ?"""

    ALL_TOM_PARAMS = []
    # Agents with a fixed "personality type":
    for prob_greedy in range(2):
        for prob_obs_other in range(2):
            ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                                   'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0,
                                   'COMPLIANCE_TOM': 0.9, 'PATH_TEAMWORK_TOM': 0.9, 'RAT_COEFF_TOM': 10,
                                   'PROB_PAUSING_TOM': prob_pausing_factor*0.7})
            ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                                   'RETAIN_GOALS_TOM': 0.8, 'LOOK_AHEAD_STEPS_TOM': 4,
                                   'PROB_THINKING_NOT_MOVING_TOM': 0,
                                   'COMPLIANCE_TOM': 0.1, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 2,
                                   'PROB_PAUSING_TOM': prob_pausing_factor*0.5})
            ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                                   'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0,
                                   'COMPLIANCE_TOM': 0.9, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 0.5,
                                   'PROB_PAUSING_TOM': prob_pausing_factor*0.4})
            ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                                   'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4,
                                   'PROB_THINKING_NOT_MOVING_TOM': 0.4,
                                   'COMPLIANCE_TOM': 0.5, 'PATH_TEAMWORK_TOM': 0.5, 'RAT_COEFF_TOM': 10,
                                   'PROB_PAUSING_TOM': prob_pausing_factor*0.6})
            ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                                   'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4,
                                   'PROB_THINKING_NOT_MOVING_TOM': 0.2,
                                   'COMPLIANCE_TOM': 0.1, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 5,
                                   'PROB_PAUSING_TOM': prob_pausing_factor*0.4})

    # Agents that fluctuate between different types
    values = [[0.7, 0.3], [0.3, 0.7]]
    for i in range(len(values)):
        prob_greedy, prob_obs_other = values[i]
        ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                               'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0,
                               'COMPLIANCE_TOM': 0.9, 'PATH_TEAMWORK_TOM': 0.9, 'RAT_COEFF_TOM': 10,
                               'PROB_PAUSING_TOM': prob_pausing_factor*0.7})
        ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                               'RETAIN_GOALS_TOM': 0.8, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0,
                               'COMPLIANCE_TOM': 0.1, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 2,
                               'PROB_PAUSING_TOM': prob_pausing_factor*0.5})
        ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                               'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0,
                               'COMPLIANCE_TOM': 0.9, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 0.5,
                               'PROB_PAUSING_TOM': prob_pausing_factor*0.4})
        ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                               'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0.4,
                               'COMPLIANCE_TOM': 0.5, 'PATH_TEAMWORK_TOM': 0.5, 'RAT_COEFF_TOM': 10,
                               'PROB_PAUSING_TOM': prob_pausing_factor*0.6})
        ALL_TOM_PARAMS.append({'PROB_GREEDY_TOM': prob_greedy, 'PROB_OBS_OTHER_TOM': prob_obs_other,
                               'RETAIN_GOALS_TOM': 0, 'LOOK_AHEAD_STEPS_TOM': 4, 'PROB_THINKING_NOT_MOVING_TOM': 0.2,
                               'COMPLIANCE_TOM': 0.1, 'PATH_TEAMWORK_TOM': 0.1, 'RAT_COEFF_TOM': 5,
                               'PROB_PAUSING_TOM': prob_pausing_factor*0.4})

    return ALL_TOM_PARAMS

def get_stats(scores_x3):
    """Get stats such as mean, median, range, SD"""
    stats_dict = {}
    stats_dict['median'] = np.median(scores_x3)
    stats_dict['mean'] = np.mean(scores_x3)
    stats_dict['std'] = np.std(scores_x3)
    print('SCORE STATS: ', stats_dict, '\n')
    return stats_dict

def plot_scores_dist(scores_x3, title, disp=True):
    """Plot the distribution of the scores_x3"""
    colours = ['b', 'r', 'y', 'c', 'm', 'g']
    f, ax = plt.subplots(1, 1, sharex='col', sharey='row')
    x_axis = [i for i in range(len(scores_x3))]
    ax.bar(x_axis, np.sort(scores_x3), 0.4, alpha=0.4, color=colours)
    ax.title.set_text(title)
    ax.set_ylabel('score')
    ax.set_xlim(0, len(x_axis))
    ax.grid()
    plt.tight_layout()
    if disp:
        plt.show()

def get_human_human_data_scores(layouts):
    """Load human human data, for layouts, for both train and test."""

    expert_data = []
    expert_data.append(get_human_human_trajectories(layouts, 'train'))
    expert_data.append(get_human_human_trajectories(layouts, 'test'))

    # Combine ALL the ep_returns into a single vector, "scores":
    scores = []
    for layout in layouts:
        for i in range(2):
            for j in range(len(expert_data[i][layout]['ep_lengths'])):
                # Check the ep_lengths are 1200 \pm 100."""
                assert 1100 < expert_data[i][layout]['ep_lengths'][j] < 1300, "Data trajectories have unexpected length!"
                scores.append(expert_data[i][layout]['ep_returns'][j])
    return scores

def get_stats_dict_toms(prob_pausing_factor, mlp, num_opponents, num_avg, layout):
    """Get stats dict for the toms"""

    scores_x3 = []  # x3 because the HH data has 1200 timesteps whereas we only give the TOMs 400 timesteps!

    np.random.seed(len(layout))  # We want each tom to play with the same toms for each step of the search
    # (but different for each layout), otherwise the search won't converge!

    # Make the TOM params:
    ALL_TOM_PARAMS = make_tom_pop(prob_pausing_factor)

    # Make all TOMs:
    tom_pop = []
    for this_tom_params in ALL_TOM_PARAMS:
        tom_agent = make_tom_agent(mlp)
        tom_agent.set_tom_params(None, None, [this_tom_params], tom_params_choice=0)
        tom_pop.append(tom_agent)
    num_toms = len(ALL_TOM_PARAMS) if testing != "True" else 2

    # Find score for each TOM:
    for i in range(num_toms):

        score_this_tom = 0
        tom_agent_player = tom_pop[i]

        # Play with num_opponents opponents. Pick opponents and the player index randomly
        for j in range(num_opponents):

            # Pick random opponent:
            k = np.random.randint(num_toms)
            if i != k:
                tom_agent_opponent = tom_pop[k]
            else:
                tom_agent_opponent = copy.deepcopy(tom_pop[k])

            # TODO: Probably not needed as get_rollouts resets??
            tom_agent_player.reset()
            tom_agent_opponent.reset()

            # Pick random player index:
            player_idx = np.random.randint(2)
            if player_idx == 0:
                agent_pair = AgentPair(tom_agent_player, tom_agent_opponent)
            elif player_idx == 1:
                agent_pair = AgentPair(tom_agent_opponent, tom_agent_player)

            trajs = env.get_rollouts(agent_pair, num_games=num_avg, final_state=False, display=False, info=False)
            sparse_rews = trajs["ep_returns"]
            avg_sparse_rew = np.mean(sparse_rews)

            # print('Score this pair: {}'.format(avg_sparse_rew))
            score_this_tom += avg_sparse_rew

        avg_score_this_tom = score_this_tom / num_opponents
        scores_x3.append(avg_score_this_tom * 3)
        # print('\n\nAvg score TOM{} on layout {}: {}\n\n'.format(i, layout_name, avg_score_this_tom))
    print("\nTOM with pp factor {}:".format(prob_pausing_factor))
    stats_dict = get_stats(scores_x3)
    # title = "TOM pop sp scores_x3 over all layouts"
    # plot_scores_dist(scores_x3, title)
    return stats_dict, scores_x3

if __name__ == "__main__":
    """Create a pop of (30?) TOMs, then play each TOM with itself on each layout. Then print a bunch of stats about the 
    performance of the TOMs, e.g. median score, mean score, range, SD. And plot the scores_x3."""

    parser = ArgumentParser()
    parser.add_argument("-a", "--num_avg",
                        help="number of rollouts to avg over", required=False, default=1, type=int)
    parser.add_argument("-t", "--testing",
                        help="whether we're testing or not", required=False, default="False")
    parser.add_argument("-l", "--layout", help="e.g. 'cramped_room'", required=True)
    parser.add_argument("-no", "--num_opponents", help="Number of (randomly selected) opponents to play with", required=False, type=int, default=5)
    parser.add_argument("-lr", "--learning_rate", help="LR", required=False, type=float, default=0.002)
    parser.add_argument("-ppf", "--prob_pausing_factor", default=0.3, type=float)

    args = parser.parse_args()
    num_avg, testing, layout, num_opponents, learning_rate, initial_prob_pausing_factor \
        = args.num_avg, args.testing, args.layout, args.num_opponents, args.learning_rate, args.prob_pausing_factor

    # --------------------------#
    # SETTINGS
    learning_rate = learning_rate
    horizon = 400
    stop_when_within = 0.02
    #--------------------------#

    # First find HH data stats
    HH_scores = get_human_human_data_scores([layout])
    print('\nHH:')
    stats_dict_hh = get_stats(HH_scores)

    # Stuff that can be done outside the loop:
    cook_time = 20
    start_order_list = 100 * ['any']
    layout_name = layout
    mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=start_order_list,
                                               cook_time=cook_time, rew_shaping_params=None)
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)
    env = OvercookedEnv(mdp, horizon=horizon)

    # Fix initial value for prob_pausing_factor
    prob_pausing_factor = initial_prob_pausing_factor if initial_prob_pausing_factor is not None else 0.3

    loss = np.inf
    step = 0
    start_time = time.perf_counter()

    while abs(loss) > stop_when_within*stats_dict_hh["mean"] and step < 1e2:
        # If the difference between the TOM mean and HH mean is more than 5% of the HH mean then keep searching

        # Find mean of TOMs:
        stats_dict_toms, TOM_scores_x3 = get_stats_dict_toms(prob_pausing_factor, mlp, num_opponents, num_avg, layout)

        # Find loss and step:
        loss = stats_dict_hh["mean"] - stats_dict_toms["mean"]
        prob_pausing_factor = prob_pausing_factor - learning_rate*loss
        prob_pausing_factor = 0 if prob_pausing_factor < 0 else prob_pausing_factor
        prob_pausing_factor = 1 if prob_pausing_factor > 1 else prob_pausing_factor
        step += 1
        print("Step {}: time elapsed: {}; loss: {}; new pp factor: {}; HH mean: {}; TOM mean: {}"
              .format(step, np.round(time.perf_counter() - start_time), loss, prob_pausing_factor, stats_dict_hh["mean"], stats_dict_toms["mean"]))

    print('\nFinal pp factor: {}\n'.format(prob_pausing_factor))
    print("HH stats: ", stats_dict_hh)
    plot_scores_dist(HH_scores, "HH scores", disp=False)
    print("TOM stats: ", stats_dict_toms)
    plot_scores_dist(TOM_scores_x3, "TOM scores x3", disp=False)
    plt.show()

