import time
from argparse import ArgumentParser
from human_aware_rl.ppo.ppo_pop import get_ppo_agent, make_tom_agent
from human_aware_rl.data_dir import DATA_DIR
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
    if layout == 'counter_circuit':
        return (3, 0), (4, 0)
    elif layout == 'coordination_ring':
        return (3, 0), (4, 1)

def get_layout_horizon(layout, horizon_length, test_agent):
    """Return the horizon for given layout/length of task"""
    extra_time = 0 if test_agent.__class__ is ToMModel else 0
    if extra_time != 0:
        print('>>>>>>> Extra time = {} <<<<<<<<'.format(extra_time))
    if horizon_length == 'short':
        return extra_time + 10
    elif horizon_length == 'medium':
        if layout == 'counter_circuit':
            return extra_time + 20
        elif layout == 'coordination_ring':
            return extra_time + 15
    elif horizon_length == 'long':
        if layout == 'counter_circuit':
            return extra_time + 30
        elif layout == 'coordination_ring':
            return extra_time + 25

def h_random_unusable_object(test_agent, mdp, standard_test_positions, print_info, random_tom_agent, layout, display_runs):
    """
    Test: H holds X, where X CANNOT currently be used
    Details:    X = O, D
                Starting locations in STPs
                H moves randomly (TOM with prob_random_action=1)
                X == 0: both pots cooked
                X == D: both pots empty
                R either holds nothing or the usable object
    """

    # Make the random TOM:
    other_player = random_tom_agent

    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    count_success = 0
    num_tests = 0
    subtest_successes = []

    first_pot_loc, second_pot_loc = find_pot_locations(layout)

    h_objects = ['onion', 'dish']
    for h_object in h_objects:

        if h_object == 'onion':
            pot_state = [ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))] # both cooked
            r_objects = [None, 'dish']
        elif h_object == 'dish':
            pot_state = None
            r_objects = [None, 'onion']

        for i, player_positions in enumerate(standard_test_positions):

            for r_object in r_objects:

                num_tests += 1
                if print_info:
                    print('\nObject held by h: {}; Held by r: {}; Pot state: {}; Players: {}\n'.
                                            format(h_object, r_object, pot_state, player_positions))

                #TODO: Instead make orientations random, with a random seed so it's repeatable?
                # Arbitrarily but deterministically choose orientation:
                ppo_or = orientations[i % 4]
                tom_or = orientations[(i+2) % 4]

                # Make the overcooked state:
                ppo_player_state = PlayerState(player_positions['r_loc'], ppo_or, None) if r_object is None else \
                                        PlayerState(player_positions['r_loc'], ppo_or,
                                                    held_object=ObjectState(r_object, player_positions['r_loc']))

                tom_player_state = PlayerState(player_positions['h_loc'], tom_or,
                                               held_object=ObjectState(h_object, player_positions['h_loc']))

                objects_dict = {pot_state[k].position: pot_state[k] for k in range(len(pot_state))} \
                    if pot_state != None else {} # Dictionary mapping positions (x, y) to ObjectStates

                state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                        order_list=['any']*100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
                # Check it's a valid state:
                mdp._check_valid_state(state_i)

                env = OvercookedEnv(mdp, start_state_fn=lambda : state_i)

                env.horizon = get_layout_horizon(layout, "long", test_agent)

                # Play with the tom agent from this state and record score
                agent_pair = AgentPair(test_agent, other_player)
                trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

                # Score in terms of whether the pot state changes:
                state_f = trajs["ep_observations"][0][-1]
                env.state = state_f
                if print_info:
                    print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
                    print('\nFinal state:\n{}'.format(env))
                if 'soup' in state_i.all_objects_by_type:
                    if state_i.all_objects_by_type['soup'] != state_f.all_objects_by_type['soup']:
                        if print_info:
                            print('Pot state has changed --> success!')
                        count_success += 1
                        subtest_successes.append('S')
                    else:
                        subtest_successes.append('F')
                else:
                    if 'soup' in state_f.all_objects_by_type:
                        if print_info:
                            print('Pot state has changed --> success!')
                        count_success += 1
                        subtest_successes.append('S')
                    else:
                        subtest_successes.append('F')

                if print_info:
                    print(count_success / num_tests)
                    print('Subtest successes: {}'.format(subtest_successes))

    return count_success, num_tests

def h_stationary_usable_object(test_agent, mdp, standard_test_positions, print_info, stationary_tom_agent, layout, display_runs):
    """
    Test: H stands still with X, where X can currently be used
    Details:    X = O, D, S, N (x4)
                Starting locations in standard_test_positions (STPs) (x12)
                Pots (2 pot states for each X):
                    if X=O, S, N: both pots empty or both with 1 onion
                    if X=D: both pots cooked or one cooked one empty
    """

    other_player = stationary_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    count_success = 0
    num_tests = 0

    first_pot_loc, second_pot_loc = find_pot_locations(layout)

    objects = ['onion', 'soup', 'dish', None]
    for object in objects:

        # ObjectState(name, position, state=None)... state (tuple or None): None for all objects except soups, for which
        # `state` is a tuple: (soup_type, num_items, cook_time)

        if object != 'dish':
            pot_states = [None,  # empty, empty
                          [ObjectState('soup', first_pot_loc, ('onion', 1, 0)), ObjectState('soup', second_pot_loc, ('onion', 1, 0))]]  # 1 onion each
        else:
            pot_states = [[ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))], # both cooked
                          [ObjectState('soup', second_pot_loc, ('onion', 3, 20))]]   # empty and cooked

        for pot_state in pot_states:

            for i, player_positions in enumerate(standard_test_positions):

                num_tests += 1
                if print_info:
                    print('\nObject held by h: {}; Pot state: {}; Players: {}\n'.format(object, pot_state, player_positions))

                # Arbitrarily but deterministically choose orientation:
                ppo_or = orientations[i % 4]
                tom_or = orientations[(i+2) % 4]

                # Make the overcooked state:
                ppo_player_state = PlayerState(player_positions['r_loc'], ppo_or, held_object=None)
                if object == 'soup':
                    tom_player_state = PlayerState(player_positions['h_loc'], tom_or,
                                        held_object=ObjectState(object, player_positions['h_loc'], ['onion', 3, 20]))
                elif object == None:
                    tom_player_state = PlayerState(player_positions['h_loc'], tom_or, None)
                else:
                    tom_player_state = PlayerState(player_positions['h_loc'], tom_or,
                                               held_object=ObjectState(object, player_positions['h_loc']))

                objects_dict = {pot_state[k].position: pot_state[k] for k in range(len(pot_state))} \
                    if pot_state != None else {} # Dictionary mapping positions (x, y) to ObjectStates

                state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                        order_list=['any']*100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
                # Check it's a valid state:
                mdp._check_valid_state(state_i)

                env = OvercookedEnv(mdp, start_state_fn=lambda : state_i)

                env.horizon = get_layout_horizon(layout, "long", test_agent)

                # Play with the tom agent from this state and record score
                agent_pair = AgentPair(test_agent, other_player)
                trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

                # Score in terms of whether the pot state changes:
                state_f = trajs["ep_observations"][0][-1]
                env.state = state_f
                if print_info:
                    print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
                    print('\nFinal state:\n{}'.format(env))
                if 'soup' in state_i.all_objects_by_type:
                    if state_i.all_objects_by_type['soup'] != state_f.all_objects_by_type['soup']:
                        if print_info:
                            print('Pot state has changed --> success!')
                        count_success += 1
                else:
                    if 'soup' in state_f.all_objects_by_type:
                        if print_info:
                            print('Pot state has changed --> success!')
                        count_success += 1

                if print_info:
                    print(count_success / num_tests)

    return count_success, num_tests

def r_holding_wrong_object(test_agent, mdp, standard_test_positions, print_info, median_tom_agent, layout, display_runs):
    """
    Test: R is holding the wrong object, and must drop it
    Details:Two variants:
                A) R has D when O needed (both pots empty)
                B) R has O when two Ds needed (both pots cooked)
            For both A and B:
                Starting locations in STPs
                Other player (H) is the median TOM
                H has nothing
    """

    # Test part A) R has D when O needed (both pots empty)

    other_player = median_tom_agent

    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)

    count_success = 0
    num_tests = 0

    for i, player_positions in enumerate(standard_test_positions):

        num_tests += 1
        if print_info:
            print('\nPlayers: {}\n'.format(player_positions))

        # Arbitrarily but deterministically choose orientation:
        ppo_or = orientations[i % 4]
        tom_or = orientations[(i + 2) % 4]

        # Make the overcooked state:
        ppo_player_state = PlayerState(player_positions['r_loc'], ppo_or,
                                       held_object=ObjectState('dish', player_positions['r_loc']))
        tom_player_state = PlayerState(player_positions['h_loc'], tom_or, held_object=None)
        objects_dict = {}  # Both pot states empty

        state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                  order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
        # Check it's a valid state:
        mdp._check_valid_state(state_i)

        env = OvercookedEnv(mdp, start_state_fn=lambda: state_i)
        env.horizon = get_layout_horizon(layout, "short", test_agent)

        # Play with the tom agent from this state and record score
        agent_pair = AgentPair(test_agent, other_player)
        trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

        # Final state:
        state_f = trajs["ep_observations"][0][-1]
        env.state = state_f
        if print_info:
            print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
            print('\nFinal state:\n{}'.format(env))

        # Success if R drops the dish:
        if not (state_f.players[0].has_object() and state_f.players[0].get_object().name == 'dish'):
            if print_info:
                print('PPO no longer has the dish --> success!')
            count_success += 1

        if print_info:
            print(count_success / num_tests)

    # Test part B) R has O when two Ds needed (both pots cooked):
    for i, player_positions in enumerate(standard_test_positions):

        num_tests += 1
        if print_info:
            print('\nPlayers: {}\n'.format(player_positions))

        # Arbitrarily but deterministically choose orientation:
        ppo_or = orientations[i % 4]
        tom_or = orientations[(i + 2) % 4]

        # Make the overcooked state:
        ppo_player_state = PlayerState(player_positions['r_loc'], ppo_or,
                                       held_object=ObjectState('onion', player_positions['r_loc']))
        tom_player_state = PlayerState(player_positions['h_loc'], tom_or, held_object=None)

        pot_state = [ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))]  # Both cooked
        objects_dict = {pot_state[k].position: pot_state[k] for k in range(len(pot_state))}

        state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                  order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
        # Check it's a valid state:
        mdp._check_valid_state(state_i)

        env = OvercookedEnv(mdp, start_state_fn=lambda: state_i)
        env.horizon = get_layout_horizon(layout, "short", test_agent)

        # Play with the tom agent from this state and record score
        agent_pair = AgentPair(test_agent, other_player)
        trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

        # Final state:
        state_f = trajs["ep_observations"][0][-1]
        env.state = state_f
        if print_info:
            print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
            print('\nFinal state:\n{}'.format(env))

        # Success if R drops the onion:
        if not (state_f.players[0].has_object() and state_f.players[0].get_object().name == 'onion'):
            if print_info:
                print('PPO no longer has the onion --> success!')
            count_success += 1

        if print_info:
            print(count_success / num_tests)

    return count_success, num_tests

def get_r_d_locations_list_4(layout):
    """R and Dish locations for test 4 (?)"""
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

def only_acessable_dish_on_counter(test_agent, mdp, print_info, stationary_tom_agent, layout, display_runs):
    """
    Test: H blocks dish dispenser but there’s a dish on counter
    Details:    4 different settings for R's location and the location of the dishes
                Both pots cooking
                H holding onion facing South; R holding nothing
                Success: R gets a dish or changes the pot state?
    """

    other_player = stationary_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)
    count_success = 0
    num_tests = 0
    subtest_successes = []
    pots = [ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))] # Both cooking
    h_locs_layout = {'counter_circuit': (1, 2), 'coordination_ring': (1, 2)}
    h_loc = h_locs_layout[layout]
    if layout == 'counter_circuit' or 'coordination_ring':  #TODO: For differnet maps we might need different ors
        tom_player_state = PlayerState(h_loc, (0, 1), held_object=ObjectState('onion', h_loc))

    r_d_locations_list = get_r_d_locations_list_4(layout)

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

    return count_success, num_tests

def r_in_the_way(test_agent, mdp, print_info, median_tom_agent, layout, display_runs):
    """
    Test: R is initially blocking H, who has the correct object
    Details:    X = O or D
                For CC:
                    RL=(1,1), HL=(1,2). H has X, R has nothing
                    RL=(6,1), HL=(6,2). H has X, R has nothing
                Pots: both empty for X=O; both full for X=D
                Median TOM
    """

    other_player = median_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)
    count_success = 0
    num_tests = 0

    if layout == 'counter_circuit':
        r_h_locations_list = [{'r_loc': (1, 1), 'h_loc': (1, 2)},
                            {'r_loc': (6, 1), 'h_loc': (6, 2)}]
    elif layout == 'coordination_ring':
        r_h_locations_list = [{'r_loc': (2, 1), 'h_loc': (1, 1)},
                            {'r_loc': (3, 2), 'h_loc': (3, 3)}]

    objects = ['onion', 'dish']

    for object in objects:
        for r_h_locs in r_h_locations_list:

            num_tests += 1
            if print_info:
                print('\nH object: {}; r_h_loc: {}\n'.format(object, r_h_locs))

            # Arbitrarily but deterministically choose orientation:
            ppo_or = orientations[i % 4]
            tom_or = orientations[(i + 2) % 4]

            # Make the overcooked state:
            ppo_player_state = PlayerState(r_h_locs['r_loc'], ppo_or, held_object=None)
            tom_player_state = PlayerState(r_h_locs['h_loc'], tom_or,
                                           held_object=ObjectState(object, r_h_locs['h_loc']))

            #TODO: More elegant way to do this!:
            # Pot states:
            if object == 'onion':
                objects_dict = {}  # Both pot states empty
            elif object == 'dish':
                pot_state = [ObjectState('soup', first_pot_loc, ('onion', 3, 20)),
                             ObjectState('soup', second_pot_loc, ('onion', 3, 20))]  # both cooked
                objects_dict = {pot_state[k].position: pot_state[k] for k in range(len(pot_state))}

            state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                      order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
            # Check it's a valid state:
            mdp._check_valid_state(state_i)

            env = OvercookedEnv(mdp, start_state_fn=lambda: state_i)
            env.horizon = get_layout_horizon(layout, "medium", test_agent)

            # Play with the tom agent from this state and record score
            agent_pair = AgentPair(test_agent, other_player)
            trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

            # Final state:
            state_f = trajs["ep_observations"][0][-1]
            env.state = state_f
            if print_info:
                print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
                print('\nFinal state:\n{}'.format(env))

            # Success if pot state changes:
            if 'soup' in state_i.all_objects_by_type:
                if state_i.all_objects_by_type['soup'] != state_f.all_objects_by_type['soup']:
                    if print_info:
                        print('Pot state has changed --> success!')
                    count_success += 1
            else:
                if 'soup' in state_f.all_objects_by_type:
                    if print_info:
                        print('Pot state has changed --> success!')
                    count_success += 1

            if print_info:
                print(count_success / num_tests)

    return count_success, num_tests

def both_have_onion_1_needed(test_agent, mdp, print_info, stationary_tom_agent, layout, display_runs):
    """
    Test: Both have onion, 1 needed, and H closer (8 tests)
    Details:    H is stationary, and stands next to the pot with the onion
                R locs: in each corner (x4)
                Pots: one is ready one has 2 onions <-- do both combinations (x2)
                Success: R no longer has onion
    """
    other_player = stationary_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)
    count_success = 0
    num_tests = 0

    tom_or = (0, -1)  # Always face the pot

    if layout == 'counter_circuit':
        r_locations_list = [{'r_loc': (1, 1)}, {'r_loc': (1, 3)}, {'r_loc': (6, 1)}, {'r_loc': (6, 3)}]
    elif layout == 'coordination_ring':
        r_locations_list = [{'r_loc': (1, 1)}, {'r_loc': (1, 3)}, {'r_loc': (3, 3)}]

    pot_states = [[ObjectState('soup', first_pot_loc, ('onion', 3, 20)), ObjectState('soup', second_pot_loc, ('onion', 2, 0))],
                  [ObjectState('soup', first_pot_loc, ('onion', 2, 0)), ObjectState('soup', second_pot_loc, ('onion', 3, 20))]]  # One cooked one need 1 onion

    for i, pot_state in enumerate(pot_states):
        for r_loc in r_locations_list:

            num_tests += 1
            if print_info:
                print('\nr_loc: {}; Pot state: {}\n'.format(r_loc, pot_state))

            # Arbitrarily but deterministically choose ppo's orientation:
            ppo_or = orientations[i % 4]

            if layout == 'counter_circuit':
                h_loc = (4, 1) if i == 0 else (3, 1)
            elif layout == 'coordination_ring':
                h_loc = (3, 1)

            # Make the overcooked state:
            ppo_player_state = PlayerState(r_loc['r_loc'], ppo_or, held_object=ObjectState('onion', r_loc['r_loc']))
            tom_player_state = PlayerState(h_loc, tom_or, held_object=ObjectState('onion', h_loc))

            objects_dict = {pot_state[k].position: pot_state[k] for k in range(len(pot_state))}

            state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                      order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
            # Check it's a valid state:
            mdp._check_valid_state(state_i)

            env = OvercookedEnv(mdp, start_state_fn=lambda: state_i)
            env.horizon = get_layout_horizon(layout, "short", test_agent)

            # Play with the tom agent from this state and record score
            agent_pair = AgentPair(test_agent, other_player)
            trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

            # Final state:
            state_f = trajs["ep_observations"][0][-1]
            env.state = state_f
            if print_info:
                print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
                print('\nFinal state:\n{}'.format(env))

            # Success if R not holding onion
            if not (state_f.players[0].has_object() and state_f.players[0].get_object().name == 'onion'):
                if print_info:
                    print('PPO no longer has the onion --> success!')
                count_success += 1

            if print_info:
                print(count_success / num_tests)

    return count_success, num_tests

def find_r_h_locations_9(layout):
    if layout == 'counter_circuit':
        return [{'r_loc': (3, 1), 'h_loc': (5, 1)},
                  {'r_loc': (3, 3), 'h_loc': (5, 1)},
                  {'r_loc': (3, 1), 'h_loc': (6, 2)},
                  {'r_loc': (3, 3), 'h_loc': (6, 2)}]
    elif layout == 'coordination_ring':
        return [{'r_loc': (1, 1), 'h_loc': (2, 1)},
                {'r_loc': (3, 2), 'h_loc': (3, 3)},
                {'r_loc': (1, 2), 'h_loc': (3, 3)},
                {'r_loc': (1, 3), 'h_loc': (3, 1)}]

def h_has_soup_o_needed(test_agent, mdp, print_info, median_tom_agent, layout, display_runs):
    """
    Test: H has soup. O needed (x8 tests)
    Details:
        For CC:
            • HL=(5,1), RL=(3,1) and (3,3)  x2
            • HL=(6,2), RL=(3,3) and (3,1)  x2
        • H = median TOM
        • Pots: both empty; one with 1 one with 2 (x2)
        • Success: Either pot state changes (We’re assuming that H will indeed deliver the soup!)
    """
    other_player = median_tom_agent
    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    first_pot_loc, second_pot_loc = find_pot_locations(layout)
    count_success = 0
    num_tests = 0

    r_h_locations_list = find_r_h_locations_9(layout)

    pot_states = [[ObjectState('soup', first_pot_loc, ('onion', 2, 0)), ObjectState('soup', second_pot_loc, ('onion', 1, 0))], None]

    for i, pot_state in enumerate(pot_states):
        for r_h_locs in r_h_locations_list:

            num_tests += 1
            if print_info:
                print('\nr_h_locs: {}; Pot state: {}\n'.format(r_h_locs, pot_state))

            # Arbitrarily but deterministically choose orientation:
            ppo_or = orientations[i % 4]
            tom_or = orientations[(i + 2) % 4]

            # Make the overcooked state:
            ppo_player_state = PlayerState(r_h_locs['r_loc'], ppo_or, held_object=None)
            tom_player_state = PlayerState(r_h_locs['h_loc'], tom_or,
                                            held_object=ObjectState('soup', r_h_locs['h_loc'], ['onion', 3, 20]))
            tom_player_state_without_soup = PlayerState(r_h_locs['h_loc'], tom_or, held_object=None)

            objects_dict = {} if pot_state is None else \
                            {pot_state[k].position: pot_state[k] for k in range(len(pot_state))}

            state_i = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                      order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.
            state_i_without_tom_soup = OvercookedState(players=[ppo_player_state, tom_player_state_without_soup], objects=objects_dict,
                                      order_list=['any'] * 100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.

            # Check it's a valid state:
            mdp._check_valid_state(state_i)

            env = OvercookedEnv(mdp, start_state_fn=lambda: state_i)
            env.horizon = get_layout_horizon(layout, "long", test_agent)

            # Play with the tom agent from this state and record score
            agent_pair = AgentPair(test_agent, other_player)
            trajs = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=display_runs, info=False)

            # Final state:
            state_f = trajs["ep_observations"][0][-1]
            env.state = state_f
            if print_info:
                print('\nInitial state:\n{}'.format(OvercookedEnv(mdp, start_state_fn=lambda: state_i)))
                print('\nFinal state:\n{}'.format(env))

            # Success if pot state has changed:
            if 'soup' in state_i_without_tom_soup.all_objects_by_type:
                if state_i_without_tom_soup.all_objects_by_type['soup'] != state_f.all_objects_by_type['soup']:
                    if print_info:
                        print('Pot state has changed --> success!')
                    count_success += 1
            else:
                if 'soup' in state_f.all_objects_by_type:
                    if print_info:
                        print('Pot state has changed --> success!')
                    count_success += 1

            if print_info:
                print(count_success / num_tests)

    return count_success, num_tests


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

def make_cc_standard_test_positions():
    # Make the standard_test_positions for this layout:
    standard_test_positions = []
    # Middle positions:
    standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (4, 1)})
    standard_test_positions.append({'r_loc': (4, 1), 'h_loc': (3, 1)})
    standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (3, 3)})
    standard_test_positions.append({'r_loc': (3, 3), 'h_loc': (3, 1)})
    # Side positions:
    standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (1, 3)})
    standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (1, 1)})
    standard_test_positions.append({'r_loc': (6, 1), 'h_loc': (6, 3)})
    standard_test_positions.append({'r_loc': (6, 3), 'h_loc': (6, 1)})
    # Diagonal positions:
    standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (6, 3)})
    standard_test_positions.append({'r_loc': (6, 3), 'h_loc': (1, 1)})
    standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (6, 1)})
    standard_test_positions.append({'r_loc': (6, 1), 'h_loc': (1, 3)})
    return standard_test_positions

def make_cring_standard_test_positions():
    # Make the standard_test_positions for CRING:
    standard_test_positions = []
    # top-R / bottom-L:
    standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (3, 2)})
    standard_test_positions.append({'r_loc': (3, 1), 'h_loc': (2, 1)})
    standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (1, 1)})
    standard_test_positions.append({'r_loc': (1, 3), 'h_loc': (3, 3)})
    # Both near dish/soup:
    standard_test_positions.append({'r_loc': (1, 2), 'h_loc': (3, 3)})
    standard_test_positions.append({'r_loc': (2, 3), 'h_loc': (1, 1)})
    # Diagonal:
    standard_test_positions.append({'r_loc': (1, 1), 'h_loc': (3, 3)})
    standard_test_positions.append({'r_loc': (3, 3), 'h_loc': (1, 1)})
    return standard_test_positions

def make_test_tom_agent(layout, mlp, tom_num):
    """Make a TOM from the validation set used for ppo"""
    VAL_TOM_PARAMS, _, _ = import_manual_tom_params(layout, 20)
    tom_agent = make_tom_agent(mlp)
    tom_agent.set_tom_params(None, None, VAL_TOM_PARAMS, tom_params_choice=int(tom_num))

    print('>>> Manually overwriting the TOM with an "optimal" TOM <<<')
    tom_agent.prob_greedy = 1
    tom_agent.prob_pausing = 0
    tom_agent.prob_random_action = 0
    tom_agent.rationality_coefficient = 20
    tom_agent.path_teamwork = 1
    tom_agent.prob_obs_other = 0
    tom_agent.wrong_decisions = 0
    tom_agent.prob_thinking_not_moving = 0
    tom_agent.look_ahead_steps = 4
    tom_agent.retain_goals = 0
    tom_agent.compliance = 0

    return tom_agent

def run_tests(layout, test_agent, tests_to_run, print_info, num_avg, mdp, mlp, display_runs):
    """..."""

    # Make TOM test agent:
    if test_agent.__class__ is str and test_agent[:3] == 'tom':
        test_agent = make_test_tom_agent(layout, mlp, tom_num=test_agent[3])
        test_agent.prob_pausing = 0
        print('>>>>>>>>>>>>>> Prob pausing = {} <<<<<<<<<<<<<<<'.format(test_agent.prob_pausing))

    # Make the TOM agents used for testing:
    stationary_tom_agent = make_stationary_tom_agent(mlp)
    median_tom_agent = make_median_tom_agent(mlp, layout)
    random_tom_agent = make_random_tom_agent(mlp, layout)

    if layout == 'counter_circuit':
        standard_test_positions = make_cc_standard_test_positions()
    elif layout == 'coordination_ring':
        standard_test_positions = make_cring_standard_test_positions()

    percent_success = [None]*10

    if "1" in tests_to_run or tests_to_run == "all":
        # TEST 1: "H stands still with X, where X CANNOT currently be used"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = h_random_unusable_object(test_agent, mdp, standard_test_positions,
                                                                print_info, random_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[1] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[1] = num_tests

    if "2" in tests_to_run or tests_to_run == "all":
        # TEST 2: "H stands still with X, where X can currently be used"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = h_stationary_usable_object(test_agent, mdp, standard_test_positions,
                                                                print_info, stationary_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[2] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[2] = num_tests

    if "3" in tests_to_run or tests_to_run == "all":
        # TEST 3: ?"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = r_holding_wrong_object(test_agent, mdp, standard_test_positions,
                                                               print_info, median_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[3] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[3] = num_tests

    if "4" in tests_to_run or tests_to_run == "all":
        # TEST 4: "H blocks dish dispenser but there’s a dish on counter"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = only_acessable_dish_on_counter(test_agent, mdp, print_info,
                                                                  stationary_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[4] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[4] = num_tests

    if "5" in tests_to_run or tests_to_run == "all":
        # TEST 5: "R is initially blocking H, who has onion or dish"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = r_in_the_way(test_agent, mdp, print_info, median_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[5] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[5] = num_tests

    if "8" in tests_to_run or tests_to_run == "all":
        # TEST 8: "Both have onion, 1 needed, and H closer"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = both_have_onion_1_needed(test_agent, mdp, print_info, stationary_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[8] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[8] = num_tests

    if "9" in tests_to_run or tests_to_run == "all":
        # TEST 9: "H has soup. O needed"
        count_successes = []
        for _ in range(num_avg):
            count_success, num_tests = h_has_soup_o_needed(test_agent, mdp, print_info, median_tom_agent, layout, display_runs)
            count_successes.append(count_success)
        percent_success[9] = round(100 * np.mean(count_successes) / num_tests)
        # num_tests_all[9] = num_tests

    print('RESULT: {}'.format(percent_success))
    # print('Num tests in each: {}'.format(num_tests_all))

    return percent_success


def plot_results(avg_dict, shorten=False):

    y_pos = np.arange(len(avg_dict.keys()))
    colour = ['B' if i % 2 == 0 else 'R' for i in range(12)]
    plt.bar(y_pos, avg_dict.values(), align='center', alpha=0.5, color=colour)
    avg_dict_keys = [list(avg_dict.keys())[i][0:6] for i in range(len(avg_dict))] if shorten else list(avg_dict.keys())
    plt.xticks(y_pos, avg_dict_keys, rotation=30)
    plt.ylabel('Avg % success')
    # plt.title('')
    plt.show()

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

def make_plot_weighted_avg_dict(run_names, results, bests, seeds):
    i = 0
    weighted_avg_dict = {}
    weighting = [0] + [2] * 3 + [1] * 2 + [0] * 2 + [1] * 2  # Give extra weight to tests 1-3 because each has many more sub-tests than the rest, and it would've made sense to split them up
    for j, run_name in enumerate(run_names):
        for seed in seeds[j]:
            for best in bests:
                b = 'V' if best == 'val' else 'T'
                this_avg = np.sum([results[i][k]*weighting[k] for k in range(len(results[i])) if results[i][k] != None]) \
                                            / np.sum(weighting)
                weighted_avg_dict['{}_{}_{}'.format(run_name, b, seed)] = this_avg
                i += 1
    # plot_results(weighted_avg_dict, shorten=True)
    return weighted_avg_dict

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

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected')

if __name__ == "__main__":
    """
    Run a qualitative experiment to test robustness of a trained agent. This code works through a suite of tests,
    largely involving putting the test-subject-agent in a specific state, then seeing if they can still play Overcooked
    from that position.
    """
    parser = ArgumentParser()
    parser.add_argument("-l", "--layout",
                        help="layout", required=False, default="counter_circuit")
    # parser.add_argument("-tm", "--time_limit", default=30, type=float)
    # parser.add_argument("-m", "--model_dir", required=False, type=str, help="...")
    parser.add_argument("-m", "--model", required=False, type=str,
                        help="e.g. 'best_models/tom_pop_expt/cc_1_tom/'", default='set_in_code')
    parser.add_argument("-s", "--seed", default=0)
    parser.add_argument("-t", "--tests_to_run", default="all")
    parser.add_argument("-pr", "--print_info", required=False, type=str, default=False)
    parser.add_argument("-dr", "--display_runs", required=False, type=str, default=False)
    parser.add_argument("-pl", "--final_plot")
    parser.add_argument("-a", "--num_avg", type=int, required=False, default=1)
    parser.add_argument("-f", "--agent_from", type=str, required=True, help='e.g. val_expt or neurips or toms')
    parser.add_argument("-r", "--run_on", required=False, type=str,
                        help="e.g. server or local", default='local')

    args = parser.parse_args()
    layout, model, seed, tests_to_run, print_info, num_avg, agent_from, final_plot, run_on, display_runs = \
        args.layout, args.model, args.seed, args.tests_to_run, str2bool(args.print_info), args.num_avg, \
        args.agent_from, args.final_plot, args.run_on, str2bool(args.display_runs)

    if model == 'set_in_code':
        # -------- Choose agents ---------
        if layout == 'counter_circuit':
            if agent_from == 'val_expt':
                run_folder = 'val_expt_cc2'
                run_names = ['cc_1_mantom', 'cc_20_mantoms', 'cc_1_bc', 'cc_20_bcs', 'cc_20_mixed']
                seeds = [[2732, 3264, 9845], [2732, 3264, 9845], [2732, 3264, 9845], [2732, 3264, 9845], [2732, 3264]]
                bests = ['train', 'val']
                shorten = False
            elif agent_from == 'neurips':
                # From Neurips paper (random3 == cc):
                run_names = ['ppo_sp_random3', 'ppo_bc_test_random3', 'ppo_hm_random3']
                seeds = [[386], [2888], [8355]]  # <-- these are the best (??) seed of each. 1st seed of each: [386, 184, 1352]
                run_folder = 'agents_neurips_paper'
                bests = [True]
                shorten = True
        elif layout == 'coordination_ring':
            if agent_from == 'neurips':
                # From Neurips paper (random1 == cring):
                run_names = ['ppo_sp_random1', 'ppo_bc_train_random1', 'ppo_bc_test_random1', 'ppo_hm_random1']
                # seeds = [[386, 2229, 7225, 7649, 9807], [516, 1887, 5578, 5987, 9456], [184, 2888, 4467, 7360, 7424], [1352, 3325, 5748, 8355, 8611]]
                seeds = [[386], [9456], [2888], [8611]]  # <-- Best seed of each?
                run_folder = 'agents_neurips_paper'
                bests = [True]
                shorten = True
        if agent_from == 'toms':
            num_toms = 1
            run_names = ['tom{}'.format(i) for i in range(num_toms)]
            seeds, bests, shorten, run_folder = [[None]]*num_toms, [None], False, ''

        if run_on == 'server':
            DIR = '/home/paul/research/human_ai_robustness/human_ai_robustness/data/ppo_runs/' + run_folder
        elif run_on == 'local':
            DIR = '/home/pmzpk/Documents/hr_coordination_from_server_ONEDRIVE/' + run_folder \
                if agent_from != 'toms' else ''
        mdp, mlp = make_mdp_mlp(layout)
        agents = []
        results = []
        for i, run_name in enumerate(run_names):

            EXPT_DIR = DIR + '/' + run_name + '/'

            for seed in seeds[i]:

                for best in bests:

                    if agent_from != 'toms':
                        test_agent, _ = get_ppo_agent(EXPT_DIR, seed, best=best)
                    else:
                        test_agent = run_name
                    # agents.append(run_name + ' >> seed_' + str(seed) + ' >> ' + best)
                    print('\n' + run_name + ' >> seed_' + str(seed) + ' >> ' + str(best))
                    results.append(run_tests(layout, test_agent, tests_to_run, print_info, num_avg, mdp, mlp, display_runs))
        # for i in range(len(agents)):
        #     print('\n#------ Full results ------#')
        #     print("Agent: {}; Results: {}\n".format(agents[i], results[i]))
        avg_dict = make_average_dict(run_names, results, bests, seeds)
        if final_plot is True:
            plot_results(avg_dict, shorten)
        weighted_avg_dic = make_plot_weighted_avg_dict(run_names, results, bests, seeds)
        save_results(avg_dict, weighted_avg_dic, results, run_folder, layout)
        print('Final average dict: {}'.format(avg_dict))
        print('Final wegihted avg: {}'.format(weighted_avg_dic))

    else:
        # Load agent to be tested:
        EXPT_DIR = DATA_DIR + 'actual_experiments/' + model
        ppo_agent, _ = get_ppo_agent(EXPT_DIR, seed, best=True)
        result = run_tests(layout, ppo_agent, tests_to_run)
