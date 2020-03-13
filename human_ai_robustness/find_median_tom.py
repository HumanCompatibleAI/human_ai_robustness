
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import copy
import numpy as np
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from human_aware_rl.ppo.ppo_pop import make_tom_agent
from human_ai_robustness.import_person_params import import_manual_tom_params

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

if __name__ == "__main__":
    """Find the median TOM (in terms of avg performance with all other TOMs)"""


    # Variables to change
    layout_code = 'cc'
    NUM_GAMES = 3
    NUM_TOMS = 30
    testing = False

    print('Layout: {}; Games: {}; Num toms: {}'.format(layout_code, NUM_GAMES, NUM_TOMS))

    #
    if layout_code == 'aa':
        layout_name = 'asymmetric_advantages'
    elif layout_code == 'croom':
        layout_name = 'cramped_room'
    elif layout_code == 'cring':
        layout_name = 'coordination_ring'
    elif layout_code == 'cc':
        layout_name = 'counter_circuit'
    else:
        raise ValueError('layout not recognised')

    cook_time = 20
    start_order_list = 100 * ['any']

    mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=start_order_list,
                                               cook_time=cook_time, rew_shaping_params=None)
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)
    env = OvercookedEnv(mdp, horizon=400)

    # Make all TOMs:
    _, _, ALL_TOM_PARAMS = import_manual_tom_params()
    tom_pop = []
    assert NUM_TOMS == len(ALL_TOM_PARAMS)
    for this_tom_params in ALL_TOM_PARAMS:
        tom_agent = make_tom_agent(mlp)
        tom_agent.set_tom_params(None, None, [this_tom_params], tom_params_choice=0)
        tom_pop.append(tom_agent)

    scores = []

    NUM_TOMS_USED = NUM_TOMS if not testing else 4

    # Find score for each TOM:
    for i in range(NUM_TOMS_USED):

        score_this_tom = 0
        tom_agent_player = tom_pop[i]

        # Play with each TOM:
        for j in range(NUM_TOMS_USED):

            if i != j:
                tom_agent_opponent = tom_pop[j]
            else:
                tom_agent_opponent = copy.deepcopy(tom_pop[j])

            # Play with each index:
            for player_idx in range(2):

                print("Agent {} playing with agent {}, index {}".format(i, j, player_idx))

                # Probably not needed as get_rollouts resets??
                tom_agent_player.reset()
                tom_agent_opponent.reset()

                if player_idx == 0:
                    agent_pair = AgentPair(tom_agent_player, tom_agent_opponent)
                elif player_idx == 1:
                    agent_pair = AgentPair(tom_agent_opponent, tom_agent_player)

                trajs = env.get_rollouts(agent_pair, num_games=NUM_GAMES, final_state=False, display=False)
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)

                print('Score this pair: {}'.format(avg_sparse_rew))

                # View if the score was exceptional:
                if avg_sparse_rew > 140 or avg_sparse_rew < 40:
                    env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True, display_until=100)
                    print('Player agent params: {}'.format(ALL_TOM_PARAMS[i]))
                    print('Opponent agent params: {}'.format(ALL_TOM_PARAMS[j]))

                score_this_tom += avg_sparse_rew

        avg_score_this_tom = score_this_tom / (NUM_TOMS_USED*2)  # x2 because 2 indices
        scores.append(avg_score_this_tom)
        print('\n\n\nAvg score this TOM: {}\n\n\n'.format(avg_score_this_tom))

    winning_score = max(scores)
    winning_tom = np.argmax(scores)

    # Find median:
    scores.append(np.inf)  # Add inf to the scores. Then there will be an odd number of scores, and the median will be a value from the set (basically we take the next score from the median!)
    median_score = np.median(scores)
    scores.pop(len(scores) - 1)  # Remove inf
    median_tom = scores.index(median_score)

    print("All scores: {}".format(scores))
    print("\nWinning tom params: {}".format(ALL_TOM_PARAMS[winning_tom]))
    print("Layout: {}; Winning_score: {}; Winning_tom: {}".format(
                                                layout_code, winning_score, winning_tom))
    print("\nMedian tom params: {}".format(ALL_TOM_PARAMS[median_tom]))
    print("Median_score: {}; Median_tom: {}".format(median_score, median_tom))
