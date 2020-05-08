
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
from human.process_dataframes import get_human_human_trajectories

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
    print('\nSCORE STATS: ', stats_dict, '\n')
    return stats_dict

def plot_scores_dist(scores_x3, title):
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

if __name__ == "__main__":
    """Create a pop of (30?) TOMs, then play each TOM with itself on each layout. Then print a bunch of stats about the 
    performance of the TOMs, e.g. median score, mean score, range, SD. And plot the scores_x3."""

    parser = ArgumentParser()
    parser.add_argument("-a", "--num_avg",
                        help="number of rollouts to avg over", required=False, default=1, type=int)
    parser.add_argument("-t", "--testing",
                        help="whether we're testing or not", required=False, default="False")
    parser.add_argument("-hd", "--hh_data",
                        help="Get stats for human-human data, instead of generating TOM data",
                        required=False, default="False")
    parser.add_argument("-hz", "--horizon", help="Game horizon", required=False, default=1200, type=int)
    parser.add_argument("-pf", "--prob_pausing_factor", help="Factor to adjust the param prob_pausing by. E.g. if 0.5 then "
        "all prob_pausing values will be half of the default value (which is in import_person_params)", required=False, default=1, type=float)

    args = parser.parse_args()
    num_avg, testing, hh_data, horizon, prob_pausing_factor = args.num_avg, args.testing, args.hh_data, args.horizon, args.prob_pausing_factor

    # ===================================#
    # Variables to change
    layouts = ['cramped_room', 'asymmetric_advantages', 'coordination_ring', 'counter_circuit']
    # ===================================#

    if not hh_data == "True":
        # Generate TOM data then get stats on the performance

        # Make the TOM params:
        ALL_TOM_PARAMS = make_tom_pop(prob_pausing_factor)

        # Stuff that can be done outside the loop:
        cook_time = 20
        start_order_list = 100 * ['any']

        scores_x3 = []

        # Loop over layouts:
        for layout_name in layouts:

            print('Layout: {}'.format(layout_name))

            mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=start_order_list,
                                                       cook_time=cook_time, rew_shaping_params=None)
            no_counters_params['counter_drop'] = mdp.get_counter_locations()
            no_counters_params['counter_goals'] = mdp.get_counter_locations()
            mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)
            env = OvercookedEnv(mdp, horizon=horizon)

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
    
                # Play with each TOM:
                for j in range(num_toms):
    
                    if i != j:
                        tom_agent_opponent = tom_pop[j]
                    else:
                        tom_agent_opponent = copy.deepcopy(tom_pop[j])
    
                    # Play with each index:
                    for player_idx in range(2):
    
                        # print("Agent {} playing with agent {}, index {}".format(i, j, player_idx))

                        # Probably not needed as get_rollouts resets??
                        tom_agent_player.reset()
                        tom_agent_opponent.reset()
    
                        if player_idx == 0:
                            agent_pair = AgentPair(tom_agent_player, tom_agent_opponent)
                        elif player_idx == 1:
                            agent_pair = AgentPair(tom_agent_opponent, tom_agent_player)
    
                        trajs = env.get_rollouts(agent_pair, num_games=num_avg, final_state=False, display=False)
                        sparse_rews = trajs["ep_returns"]
                        avg_sparse_rew = np.mean(sparse_rews)
    
                        print('Score this pair: {}'.format(avg_sparse_rew))
                        #
                        # # View if the score was exceptional:
                        # if avg_sparse_rew > 300 or avg_sparse_rew < 20:
                        #     env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True, display_until=100)
                        #     print('Player agent params: {}'.format(ALL_TOM_PARAMS[i]))
                        #     print('Opponent agent params: {}'.format(ALL_TOM_PARAMS[j]))
                        #
                        score_this_tom += avg_sparse_rew
    
                avg_score_this_tom = score_this_tom / (num_toms * 2)  # x2 because 2 indices
                scores_x3.append(avg_score_this_tom*3)

        title = "TOM pop sp scores_x3 over all layouts"

    elif hh_data == "True":
        # Get human-human data
        scores = get_human_human_data_scores(layouts)
        title = "H+H data scores_x3 over all layouts"

    else:
        raise ValueError

    stats_dict = get_stats(scores_x3)
    plot_scores_dist(scores_x3, title)
