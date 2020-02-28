import time
from argparse import ArgumentParser
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved

from human_ai_robustness.import_person_params import import_person_params
from human_aware_rl.ppo.ppo_pop import make_tom_agent, find_best_seed
from human_ai_robustness.pbt_hms import ToMAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from human_ai_robustness.agent import GreedyHumanModel_pk, ToMModel
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import logging
import numpy as np
from collections import Counter
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.utils import create_dir_if_not_exists
# np.seterr(divide='ignore', invalid='ignore')  # Suppress error about diving by zero

"""
For all states in the human data, measure how many of the TOMs in their pop and how many BCs in their pop agree with 
each other on what action to take.
"""

#------------- Helper functions for diveristy test ------------#

def make_tom_pop(mlp, tom_params, num_toms):
    """Make a population of TOM agents, with params given by tom_params"""
    tom_pop = [make_tom_agent(mlp) for i in range(num_toms)]
    for i in range(num_toms):
        tom_pop[i].set_tom_params(num_toms, None, tom_params, tom_params_choice=i)
    return tom_pop
    # return [make_tom_agent(mlp).set_tom_params(len(tom_params), None, tom_params, tom_params_choice=i) for i in range(len(tom_params))]

def find_actions_for_each_state(expert_trajs, agent, num_ep_to_use):
    """Find the actions chosen by agent in each state in the expert_trajs"""
    non_zero_actions = []
    actions_from_data = expert_trajs['ep_actions']
    num_ep_to_use = len(expert_trajs['ep_observations']) if num_ep_to_use is "all" else num_ep_to_use

    # Force the agent to act:
    if agent.human_model:
        agent.prob_pausing = 0
    else:
        agent.no_waits = True

    # For each episode:
    for i in range(num_ep_to_use):

        agent.set_agent_index(expert_trajs['metadatas']['ep_agent_idxs'][i])
        agent.reset()

        # For each state in the episode trajectory:
        for j in range(len(actions_from_data[i])):

            # Only consider states when the data also acts (otherwise we will get e.g. 5 states in a row where the data does nothing, so the state is the same, and if all toms agree on the first then they'll likely agree on the next 4!
            if actions_from_data[i][j] != (0, 0):

                current_state = expert_trajs['ep_observations'][i][j]
                # The state seems to be missing an order list. Manually add the start_order_list:
                current_state.order_list = agent.mdp.start_order_list
                # TODO: Fix this properly

                # Choose action from state
                non_zero_action, _ = agent.action(current_state)
                # Note: for TOM this also automatically updates agent.timesteps_stuck, agent.dont_drop,
                # agent.prev_motion_goal, agent.prev_state... for BC presumably this updates the history!

                if agent.human_model:
                    # Set the prev action from the data, but only if there's already a motion goal
                    if agent.prev_motion_goal != None:
                        agent.prev_best_action = actions_from_data[i][j]
                else:
                    pass  # For BC the history only stores the states, which will happen automatically

                # Make one long list of actions for all episodes:
                non_zero_actions.append(non_zero_action)

        print('Eps done: {}'.format(i))

    return non_zero_actions

def most_frequent(List):
    return max(set(List), key = List.count)

def find_agreement_of_actions(all_actions):
    """Given the actions all_actions, find how many actions are the same and add up how many states have agreement of n
    actions"""

    # NOTE: The index for count_when_n_or_more_agree corresponds to how many agree, so the 0th element should give zero (it's impossible that none agree!)
    count_when_n_agree = [0]*(len(all_actions)+1)

    # For each state:
    for i in range(len(all_actions[0])):

        actions_this_state = []

        for j in range(len(all_actions)):

            actions_this_state.append(all_actions[j][i])

        most_frequent_action = most_frequent(actions_this_state)
        number_occurances = actions_this_state.count(most_frequent_action)
        count_when_n_agree[number_occurances] += 1

    return count_when_n_agree

def find_how_many_agents_agree(expert_trajs, num_ep_to_use, population):
    """Given a tom_pop and/or bc_pop, and the set of states in the human data given by expert_trajs, quantify how much
    the agents in the pop agree on which action to take."""
    all_actions = []
    for i, agent in enumerate(population):
        all_actions.append(find_actions_for_each_state(expert_trajs, agent, num_ep_to_use))
        print('Agents done: {}'.format(i))
    assert [len(all_actions[0]) == len(all_actions[i]) for i in range(len(all_actions))]
    results = find_agreement_of_actions(all_actions)

    return results

def accumu(lis):
    total = 0
    for x in lis:
        total += x
        yield total

def process_results(count_when_n_or_more_agree):

    accumulative_count = list(reversed(list(accumu(list(reversed(count_when_n_or_more_agree))))))
    accumulative_percentage = [round(100*accumulative_count[i]/sum(count_when_n_or_more_agree))
                                                            for i in range(len(accumulative_count))]
    print('Final result: {}'.format(accumulative_percentage))
    # PLOT:
    import matplotlib.pyplot as plt;
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    plt.bar(range(1, len(accumulative_percentage)), accumulative_percentage[1:], align='center', alpha=0.5)
    plt.xticks(range(1, len(accumulative_percentage)))
    plt.ylabel('% states when at least n agents agree on the action')
    plt.xlabel('n')
    plt.title('')
    plt.show()


def make_generic_mdp_mlp(train_mdps):

    # Need some params to create TOM agent:
    LAYOUT_NAME=train_mdps[0]
    START_ORDER_LIST = ["any"] * 20

    REW_SHAPING_PARAMS = {
        "PLACEMENT_IN_POT_REW": 3,
        "DISH_PICKUP_REWARD": 3,
        "SOUP_PICKUP_REWARD": 5,
        "DISH_DISP_DISTANCE_REW": 0.015,
        "POT_DISTANCE_REW": 0.03,
        "SOUP_DISTANCE_REW": 0.1,
    }
    MDP_PARAMS = {"layout_name": LAYOUT_NAME,
                  "start_order_list": START_ORDER_LIST,
                  "rew_shaping_params": REW_SHAPING_PARAMS}
    # NO_COUNTER_PARAMS:
    START_ORIENTATIONS = False
    WAIT_ALLOWED = False
    COUNTER_PICKUP = []
    SAME_MOTION_GOALS = True

    params = {
        "MDP_PARAMS": MDP_PARAMS,
        "START_ORIENTATIONS": START_ORIENTATIONS,
        "WAIT_ALLOWED": WAIT_ALLOWED,
        "COUNTER_PICKUP": COUNTER_PICKUP,
        "SAME_MOTION_GOALS": SAME_MOTION_GOALS,
        "PERSON_PARAMScheck": None,
        "PROB_RANDOM_ACTION": 0.06  # This is needed because during metropolis sampling we assume that there is
        # always >0.01 chance of taking each action -- so our agent needs to reflect this
    }  # Using same format as pbt_toms_v2

    mdp = OvercookedGridworld.from_layout_name(**params["MDP_PARAMS"])
    # Make the mlp:
    NO_COUNTERS_PARAMS = {
        'start_orientations': START_ORIENTATIONS,
        'wait_allowed': WAIT_ALLOWED,
        'counter_goals': mdp.get_counter_locations(),
        'counter_drop': mdp.get_counter_locations(),
        'counter_pickup': COUNTER_PICKUP,
        'same_motion_goals': params["SAME_MOTION_GOALS"]
    }  # This means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, NO_COUNTERS_PARAMS, force_compute=False)

    return mdp, mlp

def make_bc_pop(num_bcs, BC_SEEDS, BC_LOCAL_DIR, layout, mdp):
    """Load each BC in the pop"""
    bc_pop = []
    for i in range(num_bcs):
        bc_name = layout + "_bc_train_seed{}".format(BC_SEEDS[i])
        print("LOADING BC MODEL FROM: {}{}".format(BC_LOCAL_DIR, bc_name))
        bc_agent, bc_params = get_bc_agent_from_saved(bc_name, unblock_if_stuck=True, stochastic=True,
                                                      overwrite_bc_save_dir=BC_LOCAL_DIR)
        bc_agent.set_mdp(mdp)
        bc_agent.human_model = False
        bc_pop.append(bc_agent)
    return bc_pop

#------------- main -----------------#

if __name__ == "__main__":
    """

    """
    parser = ArgumentParser()
    parser.add_argument("-l", "--layout",
                        help="Layout, (Choose from: cramped_room etc)",
                        required=True)
    parser.add_argument("-a", "--agent_type", help="Choose from 'tom' or 'bc' or 'both'", required=True)
    # parser.add_argument("-p", "--params", help="Starting params (all params get this value). OR set to 9 to get "
    #                                            "random values for the starting params", required=False,
    #                     default=None, type=float)
    # parser.add_argument("-ne", "--num_ep", help="Number of episodes to use when training (up to 16?)",
    #                     required=False, default=16, type=int)
    #                     help="Should make extra sure that the random search direction is not biased towards corners "
    #                          "of the hypercube.", required=False, default=True, type=bool)

    args = parser.parse_args()
    layout, agent_type = args.layout, args.agent_type

    # Settings
    num_toms = 4  # Generally set to 20
    num_bcs = 4
    num_ep_to_use = 2  # Set to "all" to use all episodes

    tom_params = import_person_params(layout, num_toms)
    BC_SEEDS = find_best_seed(layout) if num_bcs == 1 else [8502, 7786, 9094, 7709, 103, 5048, 630, 7900, 5309,
                                                8417, 862, 6459, 3459, 1047, 3759, 3806, 8413, 790, 7974, 9845]  # List of the seeds of all the BCs in the pop
    BC_LOCAL_DIR = '/home/pmzpk/bc_runs/'  # Directory of the BC agents (stored locally)

    # Load human data
    train_mdps = [layout]
    ordered_trajs = True
    human_ai_trajs = False
    data_path = "data/human/anonymized/clean_{}_trials.pkl".format('train')
    expert_trajs, _ = get_trajs_from_data(data_path, train_mdps, ordered_trajs,
                                                      human_ai_trajs, processed=False)

    mdp, mlp = make_generic_mdp_mlp(train_mdps)

    if agent_type == 'tom':
        population = make_tom_pop(mlp, tom_params, num_toms)
    elif agent_type == 'bc':
        population = make_bc_pop(num_bcs, BC_SEEDS, BC_LOCAL_DIR, layout, mdp)
    elif agent_type == 'both':
        tom_pop = make_tom_pop(mlp, tom_params, num_toms)
        bc_pop = make_bc_pop(num_bcs, BC_SEEDS, BC_LOCAL_DIR, layout, mdp)
        population = tom_pop + bc_pop
    else:
        raise ValueError

    count_when_n_agree = find_how_many_agents_agree(expert_trajs, num_ep_to_use, population)

    process_results(count_when_n_agree)
