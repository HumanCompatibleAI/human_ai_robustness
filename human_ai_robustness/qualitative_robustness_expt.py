
from argparse import ArgumentParser
from human_aware_rl.ppo.ppo_pop import get_ppo_agent
from human_aware_rl.data_dir import DATA_DIR
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import numpy as np

from human_ai_robustness.agent import ToMModel

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

def test_h_stationary_usable_object(layout, ppo_agent, mdp, mlp, standard_test_positions):
    """
    Test: H stands still with X, where X can currently be used
    Details:    X = O, D, S, N (x4)
                Starting locations in standard_test_positions (STPs) (x12)
                Pots (2 pot states for each X):
                    X=O, S, N: both pots empty or both with 2 onions
                    X=D: both pots cooked or one cooking one empty
    """

    # Make a TOM agent that doesn't move: (prob_pausing == 1, prob_random_action=0 (all other params are irrelevant))
    compliance, teamwork, retain_goals, wrong_decisions, prob_thinking_not_moving, path_teamwork, \
    rationality_coefficient, prob_pausing, prob_greedy, prob_obs_other, look_ahead_steps = [1] * 11
    tom_agent = ToMModel(mlp=mlp, prob_random_action=0, compliance=compliance, teamwork=teamwork,
                            retain_goals=retain_goals, wrong_decisions=wrong_decisions,
                            prob_thinking_not_moving=prob_thinking_not_moving, path_teamwork=path_teamwork,
                            rationality_coefficient=rationality_coefficient, prob_pausing=prob_pausing,
                            use_OLD_ml_action=False, prob_greedy=prob_greedy, prob_obs_other=prob_obs_other,
                            look_ahead_steps=look_ahead_steps)

    orientations = [(1, 0), (0, 1), (-1, 0), (0, -1)]

    rewards = []

    objects = ['onion', 'soup', 'dish', None]
    for object in objects:

        # ObjectState(name, position, state=None)... state (tuple or None): None for all objects except soups, for which
        # `state` is a tuple: (soup_type, num_items, cook_time)

        if object != 'dish':
            pot_states = [None,  # empty, empty
                          [ObjectState('soup', (3, 0), ('onion', 2, 0)), ObjectState('soup', (4, 0), ('onion', 2, 0))]]  # 2 items each
        else:
            pot_states = [[ObjectState('soup', (3, 0), ('onion', 3, 20)), ObjectState('soup', (4, 0), ('onion', 3, 20))], # both cooked
                          [ObjectState('soup', (4, 0), ('onion', 3, 5))]]   # empty and cooking

        for pot_state in pot_states:

            for i, player_positions in enumerate(standard_test_positions):

                # Arbitrarily but deterministically choose orientation:
                ppo_or = orientations[i % 4]
                tom_or = orientations[i+2 % 4]

                # Make the overcooked state:
                ppo_player_state = PlayerState(player_positions['r_loc'], ppo_or, held_object=None)
                tom_player_state = PlayerState(player_positions['h_loc'], tom_or,
                                               held_object=ObjectState(object, player_positions['h_loc']))
                agent_pair = AgentPair(ppo_agent, tom_agent)
                objects_dict = {pot_state[i].position: pot_state[i] for i in range(len(pot_state))} \
                    if pot_state != None else {} # Dictionary mapping positions (x, y) to ObjectStates

                state = OvercookedState(players=[ppo_player_state, tom_player_state], objects=objects_dict,
                                        order_list=['any']*100)  # players: List of PlayerStates (order corresponds to player indices). objects: Dictionary mapping positions (x, y) to ObjectStates.

                # Check it's a valid state:
                mdp._check_valid_state(state)

                env = OvercookedEnv(mdp, start_state_fn=lambda : state)

                env.horizon = 200

                # Play with the tom agent from this state and record score
                trajs = env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)
                avg_sparse_rew = np.mean(trajs["ep_returns"])
                print('Rew: {}'.format(avg_sparse_rew))
                rewards.append(avg_sparse_rew)

    return rewards

def run_tests(layout, test_agent):
    """..."""

    # Make the standard mdp for this layout:
    mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=None, cook_time=20, rew_shaping_params=None)
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

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

    # TEST 1: "H stands still with X, where X can currently be used"
    results = test_h_stationary_usable_object(layout, test_agent, mdp, mlp, standard_test_positions)

    return results



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
    parser.add_argument("-m", "--model_dir", required=False, type=str,
                        help="e.g. 'best_models/tom_pop_expt/cc_1_tom/'")
    parser.add_argument("-s", "--seed", default=0)

    args = parser.parse_args()
    layout, model_dir, seed = args.layout, args.model_dir, args.seed

    # Load agent to be tested
    EXPT_DIR = DATA_DIR + 'actual_experiments/' + model_dir
    ppo_agent, _ = get_ppo_agent(EXPT_DIR, seed, best=True)


    result = run_tests(layout, ppo_agent)