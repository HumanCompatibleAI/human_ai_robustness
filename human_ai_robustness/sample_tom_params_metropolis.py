import time
from argparse import ArgumentParser
from human_aware_rl.human.process_dataframes import get_trajs_from_data
from human_ai_robustness.pbt_hms import ToMAgent
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from human_ai_robustness.agent import GreedyHumanModel_pk
from overcooked_ai_py.planning.planners import MediumLevelPlanner
import logging
import numpy as np
from collections import Counter
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.utils import create_dir_if_not_exists
# np.seterr(divide='ignore', invalid='ignore')  # Suppress error about diving by zero

"""
Here we only optimise the TOM parameter on states for which the data acts. So if the data acts, we force the TOM to 
also act, then compare their actions.
"""

# Helper functions:

def choose_tom_actions(expert_trajs, tom_agent, num_ep_to_use):
    """
    Take a human model with given parameters, then use this to choose one action for every state in the data.
    Correction: now we only find one action for each state in which the data acts!
    Note that the TOM odel retains a memory of previous plans/actions/other info
    :return: tom_actions, a list of lists of actions chosen by the TOM
    """

    tom_actions = []
    actions_from_data = expert_trajs['ep_actions']

    # For each episode we want to use
    for i in range(num_ep_to_use):

        tom_actions_this_ep = []

        tom_agent.set_agent_index(expert_trajs['metadatas']['ep_agent_idxs'][i])
        tom_agent.reset()
        tom_agent.look_ahead_steps = int(np.round(tom_agent.look_ahead_steps))

        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # Only let the TOM take an action when the data also acts. Otherwise force TOM action to (0,0) also
            if actions_from_data[i][j] != (0,0):

                current_state = expert_trajs['ep_observations'][i][j]
                # The state seems to be missing an order list. Manually add the start_order_list:
                current_state.order_list = tom_agent.mdp.start_order_list
                #TODO: Fix this properly

                # # Print the state to view the game:
                # overcooked_env = OvercookedEnv(tom_agent.mdp)
                # logging.warning('i = {}, j = {}'.format(i, j))
                # overcooked_env.state = current_state
                # logging.warning(overcooked_env)

                # Force the agent to take an action
                temp_prob_pausing = tom_agent.prob_pausing
                tom_agent.prob_pausing = 0

                # Choose TOM action from state
                tom_action = tom_agent.action(current_state)[0]
                # This also automatically updates tom_agent.timesteps_stuck, tom_agent.dont_drop,
                # tom_agent.prev_motion_goal, tom_agent.prev_state

                # Then reset the prob_pausing:
                tom_agent.prob_pausing = temp_prob_pausing

                # Set the prev action from the data, but only if there's already a motion goal
                if tom_agent.prev_motion_goal != None:
                    tom_agent.prev_best_action = actions_from_data[i][j]

                # Print everything after:
                logging.warning('Action from TOM: {}; Action from data: {}'.format(tom_action, actions_from_data[i][j]))
                logging.warning('TOM prev_motion_goal: {}; TOM dont_drop: {}'.format(tom_agent.prev_motion_goal,
                                                                                   tom_agent.dont_drop))
                logging.warning('TOM time stuck: {}'.format(tom_agent.timesteps_stuck))

            else:
                tom_action = (0,0)

            tom_actions_this_ep.append(tom_action)

        tom_actions.append(tom_actions_this_ep)

    return tom_actions

def find_tom_probs_action_in_state(multi_tom_agent, actions_from_data, num_ep_to_use, expert_trajs):
    """
    Find the prob that tom takes action a in state s. BUT only for the actions the data actually takes (we don't need
    the probs for the other actions, as the loss is zero for these)

    :param multi_tom_agent:
    :param actions_from_data:
    :param num_ep_to_use:
    :param expert_trajs:
    :return:
    """

    multi_tom_actions = []

    # For each agent
    for tom_agent in multi_tom_agent:

        # Find all actions by this agent:
        tom_actions = choose_tom_actions(expert_trajs, tom_agent, num_ep_to_use)
        multi_tom_actions.append(tom_actions)

    # Now have all actions for all agents

    # List of lists of zeros:
    tom_probs_action_in_state = [[0] * actions_from_data[i].__len__() for i in range(num_ep_to_use)]

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # For each agent
            for tom_actions in multi_tom_actions:
            #TODO: Should really swap this "for" with the "if" below

                # Only work out the probs for states where the data acts:
                if actions_from_data[i][j] != (0,0):

                    # If agent chooses the same action as the data, then count this
                    if actions_from_data[i][j] == tom_actions[i][j]:

                        # 1 / number of agents. Therefore if all agents act correctly then prob will be 1
                        tom_probs_action_in_state[i][j] += 1/multi_tom_agent.__len__()

                # data gives (0,0):
                else:
                    # Set to -1 to signal that we're not using this probability
                    tom_probs_action_in_state[i][j] = -1
                    # Both the data and the agent should give (0,0):
                    assert actions_from_data[i][j] == tom_actions[i][j]

            # Force prob to be 0.01 minimum (otherwise we get infinities in the cross entropy):
            if tom_probs_action_in_state[i][j] == 0:
                tom_probs_action_in_state[i][j] = 0.01

    return tom_probs_action_in_state

def find_prob_not_acting(actions_from_data, num_ep_to_use):

    count_number_not_acting = 0
    count_total_states = 0

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            count_total_states += 1

            if actions_from_data[i][j] == (0,0):
                # Count for how many of the states the data doesn't act:
                count_number_not_acting += 1

    prob_data_doesnt_act = count_number_not_acting / count_total_states
    number_states_with_acting = count_total_states - count_number_not_acting

    return prob_data_doesnt_act, number_states_with_acting

def find_cross_entropy_loss(actions_from_data, expert_trajs, multi_tom_agent, num_ep_to_use):
    """
    ...?
    :param expert_trajs:
    :param multi_tom_agent:
    :param num_ep_to_use:
    :return:
    """
    # Find Prob_TOM(action|state) for all actions chosen by the data
    tom_probs_action_in_state = find_tom_probs_action_in_state(multi_tom_agent, actions_from_data, num_ep_to_use,
                                                          expert_trajs)

    loss = 0

    # For each episode we want to use
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # Only add to the loss if it's a state for which the data acts:
            if actions_from_data[i][j] != (0,0):
                loss += np.log(tom_probs_action_in_state[i][j])/np.log(0.01)  # Normalise by 0.01, so that loss_ij=1 max
            else:
                assert tom_probs_action_in_state[i][j] == -1

    return loss

def two_most_frequent(List):
    occurence_count = Counter(List)
    second_action = None
    if len(occurence_count) > 1:
        second_action = occurence_count.most_common(2)[1][0]
    return occurence_count.most_common(1)[0][0], second_action

def find_top_12_accuracy(actions_from_data, expert_trajs, multi_tom_agent, num_ep_to_use, number_states_with_acting):
    """
    Find top-1 (top-2) accuracy, which is the proportion of states in which the TOM's most likely (2nd most likely)
    action equals the action from the data
    """

    multi_tom_actions = []
    # For each agent
    for tom_agent in multi_tom_agent:
        # Find all actions by this agent:
        tom_actions = choose_tom_actions(expert_trajs, tom_agent, num_ep_to_use)
        multi_tom_actions.append(tom_actions)
    # Now have all actions for all agents

    count_top_1 = 0
    count_top_2 = 0

    # For each state:
    for i in range(num_ep_to_use):
        for j in range(actions_from_data[i].__len__()):

            # Only consider states where the data takes an action:
            if actions_from_data[i][j] != (0, 0):

                # Change format of multi_tom_actions:
                list_actions = []
                for k in range(len(multi_tom_actions)):
                    list_actions.append(multi_tom_actions[k][i][j])

                #TODO: Here we're ignoring draws between two actions:
                top_action, second_action = two_most_frequent(list_actions)

                assert top_action != second_action

                # If the top/2nd action is the same as the data, count it:
                if top_action == actions_from_data[i][j]:
                    count_top_1 += 1
                    count_top_2 += 1
                elif second_action == actions_from_data[i][j]:
                    count_top_2 += 1

    top_1_acc = count_top_1 / number_states_with_acting
    top_2_acc = count_top_2 / number_states_with_acting

    assert top_2_acc >= top_1_acc

    return top_1_acc, top_2_acc

def shift_by_epsilon(params, epsilon):
    """
    Shift the PERSON_PARAMS_TOM by epsilon, except RATIONALITY_COEFF, which isn't between 0 and 1 so is shifted further
    :return: PERSON_PARAMS_TOMeps that've been shifted
    """

    PERSON_PARAMS_TOM = params['PERSON_PARAMS_TOM']
    PERSON_PARAMS_TOMeps = params['PERSON_PARAMS_TOMeps']
    PERSON_PARAMS_FIXED = params['PERSON_PARAMS_FIXED']

    for i, pparam in enumerate(PERSON_PARAMS_TOM):

        if pparam in PERSON_PARAMS_FIXED:
            pass  # Don't change this parameter
        else:
            PERSON_PARAMS_TOMeps[pparam+'eps'] = PERSON_PARAMS_TOM[pparam] + epsilon[i]
        # print('param before: {}; param shifted raw: {}'.format(PERSON_PARAMS_TOM[pparam], PERSON_PARAMS_TOMeps[pparam+'eps']))

        # Ensure all params between 0 and 1; except RATIONALITY_COEFF which should be >= 0
        # RATIONALITY_COEFF can reasonably range from 0 to e.g. 10, so we need to shift it further!
        if pparam is 'RATIONALITY_COEFF_TOM':
            PERSON_PARAMS_TOMeps[pparam+'eps'] = PERSON_PARAMS_TOM[pparam] + epsilon[i] * 10
            if PERSON_PARAMS_TOMeps[pparam+'eps'] < 0:
                PERSON_PARAMS_TOMeps[pparam+'eps'] = 0
            else: pass
        else:
            if PERSON_PARAMS_TOMeps[pparam+'eps'] < 0:
                PERSON_PARAMS_TOMeps[pparam+'eps'] = 0
            elif PERSON_PARAMS_TOMeps[pparam+'eps'] > 1:
                PERSON_PARAMS_TOMeps[pparam+'eps'] = 1
            else: pass

        # print('param before: {}; param shifted final: {}'.
        #       format(PERSON_PARAMS_TOM[pparam], PERSON_PARAMS_TOMeps[pparam+'eps']))

    return PERSON_PARAMS_TOMeps

def shift_by_gradient(params, epsilon, delta_loss, lr):
    """
    Shift PERSON_PARAMS_TOM in the direction of negative gradient, scaled by lr
    """

    PERSON_PARAMS_TOM = params['PERSON_PARAMS_TOM']
    PERSON_PARAMS_FIXED = params['PERSON_PARAMS_FIXED']

    for i, pparam in enumerate(PERSON_PARAMS_TOM):
        # print('param before: {}'.format(PERSON_PARAMS_TOM[pparam]))
        if pparam in PERSON_PARAMS_FIXED:
            # Don't change this parameter
            pass
        else:
            PERSON_PARAMS_TOM[pparam] = PERSON_PARAMS_TOM[pparam] + epsilon[i]*delta_loss*lr
        # print('delta_loss: {}; lr: {}; this eps: {}'.format(delta_loss, lr, epsilon[i]))
        # print('param shifted raw: {}'.format(PERSON_PARAMS_TOM[pparam]))

        # Ensure all params between 0 and 1; except RATIONALITY_COEFF which should be >= 0
        # RATIONALITY_COEFF gets shifted by 10* compared to the others
        if pparam is 'RATIONALITY_COEFF_TOM':
            PERSON_PARAMS_TOM[pparam] = PERSON_PARAMS_TOM[pparam] + epsilon[i]*delta_loss*lr*10
            if PERSON_PARAMS_TOM[pparam] < 0:
                PERSON_PARAMS_TOM[pparam] = 0
            else: pass
        else:
            if PERSON_PARAMS_TOM[pparam] < 0:
                PERSON_PARAMS_TOM[pparam] = 0
            elif PERSON_PARAMS_TOM[pparam] > 1:
                PERSON_PARAMS_TOM[pparam] = 1
            else: pass

        # print('param shifted final: {}'.format(PERSON_PARAMS_TOM[pparam]))
    # return PERSON_PARAMS_TOM

def find_gradient_and_step_multi_tom(params, mlp, expert_trajs, num_ep_to_use, lr, epsilon_sd,
                                    start_time, step_number, total_number_steps):
    """
    Same as find_gradient_and_step_single_tom except here we have multiple toms taking multiple actions, so we find
    prob(action|state) and use cross entropy loss
    :return: loss
    """

    actions_from_data = expert_trajs['ep_actions']

    tom_number = ''
    # Make multiple tom agents:
    multi_tom_agent = ToMAgent(params, 99, tom_number).get_multi_agent(mlp)

    loss = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_tom_agent, num_ep_to_use)

    # Choose random epsilon (eps) from normal dist, sd=epsilon_sd
    epsilon = np.random.normal(scale=epsilon_sd, size=params['PERSON_PARAMS_TOM'].__len__())

    # Turn into random unit vector:
    if params["ensure_random_direction"]:
        print('NOTE: This only works for sd=0.01 and 8 pparams. AND I did this bit quickly so should be checked!')
        #TODO: There might be an issue that we have 8 pparams but only 6 are usually allowed to vary... this is like
        # taking a random direction then projecting onto the 6-dim subspace, which should be fine (??)

        print('ep initial length = {}'.format(np.sqrt(sum(epsilon**2))))

        # Make random vector from Gaussian sd=1
        random_gauss_vect = np.random.normal(scale=1, size=8)
        length = np.sqrt(sum(random_gauss_vect**2))
        unit_vect = random_gauss_vect / length
        # print('Unit length = {}'.format(np.sqrt(sum(unit_vect**2))))

        # Now scale by 0.023, which is the average length of an 8-dim vector from Gaussians with sd=0.01
        epsilon = 0.027*unit_vect
        print('ep final length = {}'.format(np.sqrt(sum(epsilon**2))))

    # Make PERSON_PARAMS_TOM + epsilon. For rationality_coefficient, do eps*10 for now. shift_by_epsilon also ensures all
    # params are between 0 and 1:
    shift_by_epsilon(params, epsilon)

    # Find loss for new params
    tom_number = 'eps'
    multi_tom_agent_eps = ToMAgent(params, 99, tom_number).get_multi_agent(mlp)
    loss_eps = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_tom_agent_eps, num_ep_to_use)
    delta_loss = loss - loss_eps

    # Set new personality params by shifting in the direction of downhill
    raise ValueError('Code doesnt work past here because I havent put limits on the new params from the new TOM. For '
                     'example, self.look_ahead_steps must be >1, but this limit isnt included here')
    shift_by_gradient(params, epsilon, delta_loss, lr)

    # What's the loss after this grad step:
    # tom_number = ''
    # multi_tom_agent = ToMAgent(params, 99, tom_number).get_multi_agent(mlp)
    # loss_final = find_cross_entropy_loss(actions_from_data, expert_trajs, multi_tom_agent, num_ep_to_use)
    # return loss_final

    if (step_number % (total_number_steps / 1000)) == 0:
        print('Completed {}% in time {} mins; Loss before grad step: {}'.format(100 * step_number / total_number_steps,
                                                     round((time.time() - start_time)/60), loss))
    if (step_number % (total_number_steps / 100)) == 0:
        print(params["PERSON_PARAMS_TOM"])

#--------------------------------------------#
#------------ Metropolis sampling -----------#

def iterate_metropolis_sampling(params, mlp, expert_trajs, num_ep_to_use, epsilon_sd, start_time, step_number,
                                total_number_steps, accepted_history, step_size, save_filename):
    """Randomly sample a new candidate set of params, calculate ratio of the probabilities that the new:old params
    recover the data, then accepts or reject the candidate params."""

    tom_number = ''
    multi_tom_agent_initial = ToMAgent(params, 99, tom_number).get_multi_agent(mlp)  # Make multiple tom agents

    #TODO: Note that (at the time of writing) this is proportional to the cross entropy loss!
    initial_log_prob = find_log_prob_data_given_params(expert_trajs, multi_tom_agent_initial, num_ep_to_use)

    generate_candidate_params(params, epsilon_sd, step_size)

    tom_number = 'eps'  # This means that multi_tom_agent_cand will use the params that were shifted by epsilon
    multi_tom_agent_cand = ToMAgent(params, 99, tom_number).get_multi_agent(mlp)  # Make multiple 'candidate' tom agents

    candidate_log_prob = find_log_prob_data_given_params(expert_trajs, multi_tom_agent_cand, num_ep_to_use)

    accepted = acceptance_function(initial_log_prob, candidate_log_prob)

    step_size = print_save_sampling_info(params, start_time, step_number, total_number_steps, accepted,
                                         accepted_history, step_size, save_filename)

    return step_size


def print_save_sampling_info(params, start_time, step_number, total_number_steps, accepted, accepted_history,
                             step_size, save_filename):
    """Print relevant information about the sampling algorithm and the sampled params"""

    display_steps = 10

    accepted_history[step_number % len(accepted_history)] = accepted

    if step_number % display_steps == 0:
        print('Completed {} steps in time {} mins'.format(step_number, round((time.time() - start_time)/60)))
        print(params['PERSON_PARAMS_TOM'])
        prop_accepted = np.mean(accepted_history)
        print('Proportion accepted: {}'.format(prop_accepted))
        if prop_accepted < 0.23:
            step_size -= 0.01  # If the step size was 0, then we would always accept
        else:
            step_size += 0.01
        print('New step size: {}\n'.format(step_size))

        #TODO: it would be much more elegant to do this using logging; or use helper functions in utils:
        with open(save_filename, 'a') as f:
            f.write('Completed {} steps in time {} mins. PPARAMS: {} \n'.format(step_number, round((time.time() -
                                                                    start_time)/60), str(params['PERSON_PARAMS_TOM'])))
    return step_size

def find_log_prob_data_given_params(expert_trajs, multi_tom_agent, num_ep_to_use):
    """Find the probability that the TOM with params in multi_tom_agent reproduces the data -- i.e. the prob that all
    its actions will agree with those from the data (ignoring states for which the data does a zero action).
    Return the log of the total probability"""

    # Find Prob_TOM(action|state) for all actions chosen by the data
    actions_from_data = expert_trajs['ep_actions']
    tom_probs_action_in_state = find_tom_probs_action_in_state(multi_tom_agent, actions_from_data, num_ep_to_use,
                                                               expert_trajs)

    log_prob_data_given_params = 0

    # For each episode we want to use:
    for i in range(num_ep_to_use):
        # For each state in the episode trajectory:
        for j in range(actions_from_data[i].__len__()):

            # Only states for which the data acts:
            if actions_from_data[i][j] != (0, 0):
                log_prob_data_given_params += np.log(tom_probs_action_in_state[i][j])
            else:
                assert tom_probs_action_in_state[i][j] == -1

    return log_prob_data_given_params

def acceptance_function(initial_log_prob, candidate_log_prob):
    """Determine if we are to accept or reject the new candidate params; if we accept then set params[
    'PERSON_PARAMS_TOM'] to the new params"""
    acceptance_ratio = np.exp(candidate_log_prob - initial_log_prob)
    logging.info('acceptance_ratio: {}'.format(acceptance_ratio))
    if np.random.rand() <= acceptance_ratio:
        for i, pparam in enumerate(params['PERSON_PARAMS_TOM']):
            params['PERSON_PARAMS_TOM'][pparam] = params['PERSON_PARAMS_TOMeps'][pparam + 'eps']
        return 1
    else:
        return 0

def generate_candidate_params(params, epsilon_sd, step_size):
    """First pick epsilon -- a random Gaussian vector (need a better description of this!). Then for each personality
    parameter, convert it to logit (after converting to a prob first if needed), step by epsilon, then convert back
    to a prob. This gives the new set of parameters"""

    # Vector for the random step (in logit space)
    epsilon = pick_random_step(epsilon_sd, params, step_size)

    person_params_tom = params['PERSON_PARAMS_TOM']
    person_params_tom_eps = params['PERSON_PARAMS_TOMeps']
    person_params_fixed = params['PERSON_PARAMS_FIXED']

    for i, pparam in enumerate(person_params_tom):

        if pparam in person_params_fixed:
            # Don't change these params:
            person_params_tom_eps[pparam + 'eps'] = person_params_tom[pparam]
        else:
            param_logit = convert_to_logit(person_params_tom[pparam], pparam)
            new_param_logit = param_logit + epsilon[i]
            new_param = convert_back_from_logit(new_param_logit, pparam)
            person_params_tom_eps[pparam + 'eps'] = new_param

    logging.info('Old params: {}'.format(person_params_tom))
    logging.info('New params: {}'.format(person_params_tom_eps))

def convert_to_logit(param_value, pparam):
    """First convert any parameters that aren't probs to probs; then convert to logit"""

    # Convert params to probs if they aren't already:
    if pparam == "RAT_COEFF_TOM":
        param_value = param_value/20
    elif pparam == "LOOK_AHEAD_STEPS_TOM":
        #TODO: Better to keep the parameter as a probability, and convert it to an integer only when taking actions?
        #TODO: This assumes that we only consider 2, 3, or 4 lookahead steps
        param_value = (param_value - 1.5)/3
        #TODO: Is this right? Is there a less elborate way?
    return np.log(param_value/(1-param_value))

def convert_back_from_logit(logit_value, pparam):
    """First convert back from logit to prob; then convert any parameters that aren't meant to be probs back to their original form"""

    param_value = 1/(1+np.exp(-logit_value))
    # Convert params to probs if they aren't already:
    if pparam == "RAT_COEFF_TOM":
        return param_value*20
    elif pparam == "LOOK_AHEAD_STEPS_TOM":
        return param_value*3 + 1.5
    else:
        return param_value


def pick_random_step(epsilon_sd, params, base_leaning_rate):

    # Choose random epsilon (eps) from normal dist, sd=epsilon_sd
    epsilon = np.random.normal(scale=epsilon_sd, size=len(params['PERSON_PARAMS_TOM']))

    # Turn into random unit vector:
    if params["ensure_random_direction"]:

        logging.info('Unnormed length = {}'.format(np.sqrt(sum(epsilon ** 2))))

        # Make random vector from Gaussian sd=1
        random_gauss_vect = np.random.normal(scale=1, size=len(params['PERSON_PARAMS_TOM']))
        length = np.sqrt(sum(random_gauss_vect ** 2))
        unit_vect = random_gauss_vect / length
        epsilon = step_size * unit_vect
        logging.info('ep final length = {}'.format(np.sqrt(sum(epsilon ** 2))))

    return epsilon

def find_length_gaussian_vector(sd, size):

    average_over = 1000
    a = 0
    for i in range(average_over):
        random_gauss_vect = np.random.normal(scale=sd, size=size)
        length = np.sqrt(sum(random_gauss_vect ** 2))
        a.append(length)
    return np.mean(a)

#------------- main -----------------#

if __name__ == "__main__":
    """

    """
    parser = ArgumentParser()
    parser.add_argument("-l", "--layout",
                        help="Layout, (Choose from: cramped_room etc)",
                        required=True)
    parser.add_argument("-p", "--params", help="Starting params (all params get this value). OR set to 9 to get "
                                               "random values for the starting params", required=False,
                        default=None, type=float)
    parser.add_argument("-ne", "--num_ep", help="Number of episodes to use when training (up to 16?)",
                        required=False, default=16, type=int)
    parser.add_argument("-lr", "--base_lr", help="Base learning rate. E.g. 0.1", default=0.1, type=float,
                        required=False)
    parser.add_argument("-ss", "--step_size", help="Step size for the step in the metropolis algorithm. E.g. 1.3 is "
                        "approx for a 7D Gaussian vector, sd=0.5 (see find_length_gaussian_vector)", required=False,
                        default=1.3, type=float)
    parser.add_argument("-sd", "--epsilon_sd", type=float,
                        help="Standard deviation of dist picking epison from. Initial runs suggest sd=0.02 is good",
                        required=False, default=0.02)
    parser.add_argument("-nh", "--num_toms", help="Number of human models to use for approximating P(action|state)",
                        required=False, default=3, type=int)
    parser.add_argument("-ns", "--num_grad_steps",  help="Number of gradient decent steps", required=False,
                        default=1e9, type=int)
    parser.add_argument("-t", "--run_type",
                        help="Set to met to do metropolis sampling, zeroth to do zeroth order opt, or acc to just "
                             "check the top-1 and top-2 accuracy",
                        required=False, default="met")
    # This gives problems with the BOOL for some reason:
    parser.add_argument("-r", "--ensure_random_direction",
                        help="Should make extra sure that the random search direction is not biased towards corners "
                             "of the hypercube.", required=False, default=False, type=bool)

    args = parser.parse_args()
    layout = args.layout
    starting_params = args.params

    # -----------------------------#
    # Settings for the zeroth order optimisation:
    num_ep_to_use = args.num_ep  # How many episodes to use for the fitting
    base_learning_rate = args.base_lr  # Quick test suggests loss for a single episode can be up to around 10. So if
    # base learning rate is 1/5, we shift by roughly epsilon. NEED TO TUNE THIS!
    step_size = args.step_size
    epsilon_sd = args.epsilon_sd  # standard deviation of the dist to pick epsilon from
    ensure_random_direction = args.ensure_random_direction
    number_toms = args.num_toms
    total_number_steps = args.num_grad_steps  # Number of steps to do in gradient decent
    run_type = args.run_type
    # -----------------------------#

    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)
    logging.getLogger().setLevel(logging.ERROR)  # pk: Note sure why I need this line too

    # Load human data as state-action pairs:
    # TODO: Do other layouts
    train_mdps = [layout]
    ordered_trajs = True
    human_ai_trajs = False
    data_path = "data/human/anonymized/clean_{}_trials.pkl".format('train')
    #TODO: Double check that processed should be False, in order to give states rathers than obs? Micah says "I think
    # that processed returns state encodings
    # (stacks of masks) rather than OvercookedState objects, or at least this was the original intention in the code –
    # hopefully that is still true or the code is broken (edited)":
    expert_trajs, _ = get_trajs_from_data(data_path, train_mdps, ordered_trajs,
                                                      human_ai_trajs, processed=False)
    # Load (file I saved using pickle) instead FOR SIMPLE ONLY???: pickle_in = open('expert_trajs.pkl',
    # 'rb'); expert_trajs = pickle.load(pickle_in)

    # Starting personality params
    if starting_params == None:
        raise ValueError("Not set up yet")
        # PERSON_PARAMS_TOM = {DEFAULT PARAMS???}
    elif starting_params == 9:
        # Random initialisation:
        PERSON_PARAMS_TOM = {
            "COMPLIANCE_TOM": np.random.rand(),
            "RETAIN_GOALS_TOM": np.random.rand(),
            "PATH_TEAMWORK_TOM": np.random.rand(),
            "RAT_COEFF_TOM": np.random.rand()*20,
            "PROB_GREEDY_TOM": np.random.rand(),
            "PROB_OBS_OTHER_TOM": np.random.rand(),
            "LOOK_AHEAD_STEPS_TOM": np.random.rand()*3+1.5,
            "PROB_THINK_TOM": 1,
            "PROB_PAUSING_TOM": 99  # This will be modified within the code. Setting to 99 gives an error if it's not
        }
    else:
        PERSON_PARAMS_TOM = {
            "COMPLIANCE_TOM": starting_params,
            "RETAIN_GOALS_TOM": starting_params,
            "PATH_TEAMWORK_TOM": starting_params,
            "RAT_COEFF_TOM": starting_params*20,
            "PROB_GREEDY_TOM": starting_params,
            "PROB_OBS_OTHER_TOM": starting_params,
            "LOOK_AHEAD_STEPS_TOM": starting_params*3+1.5,
            "PROB_THINK_TOM": 1,
            "PROB_PAUSING_TOM": 99  # This will be modified within the code. Setting to 99 gives an error if it's not
        }

    # Irrelevant what values these start as:
    PERSON_PARAMS_TOMeps = {
        "COMPLIANCE_TOMeps": 9,
        "RETAIN_GOALS_TOMeps": 9,
        "PATH_TEAMWORK_TOMeps": 9,
        "RAT_COEFF_TOMeps": 9,
        "PROB_GREEDY_TOMeps": 9,
        "PROB_OBS_OTHER_TOMeps": 9,
        "LOOK_AHEAD_STEPS_TOMeps": 9,
        "PROB_THINK_TOMeps": 9,
        "PROB_PAUSING_TOMeps": 9
    }

    # Keep some of the person params fixed. E.g. put {"PROB_PAUSING_TOM"}
    PERSON_PARAMS_FIXED = {"PROB_PAUSING_TOM", "PROB_THINK_TOM"}

    # Need some params to create TOM agent:
    LAYOUT_NAME = train_mdps[0]
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
        "PERSON_PARAMS_TOM": PERSON_PARAMS_TOM,
        "PERSON_PARAMS_TOMeps": PERSON_PARAMS_TOMeps,
        "PERSON_PARAMS_FIXED": PERSON_PARAMS_FIXED,
        "START_ORIENTATIONS": START_ORIENTATIONS,
        "WAIT_ALLOWED": WAIT_ALLOWED,
        "COUNTER_PICKUP": COUNTER_PICKUP,
        "SAME_MOTION_GOALS": SAME_MOTION_GOALS,
        "ensure_random_direction": ensure_random_direction,
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

    #-----------------------------#
    lr = base_learning_rate / num_ep_to_use  # learning rate: the more episodes we use the more the loss will be,
    # so we need to scale it down by num_ep_to_use
    params["sim_threads"] = number_toms  # Needed when using ToMAgent
    actions_from_data = expert_trajs['ep_actions']
    # First find the probability of the data not acting:
    prob_data_doesnt_act, number_states_with_acting = find_prob_not_acting(actions_from_data, num_ep_to_use)
    print('Prob of data-agent taking ZERO action, (0,0): {}; Number states when data-agent acts: {}'.format(
        prob_data_doesnt_act, number_states_with_acting))

    DIR = DATA_DIR + 'metropolis/'
    create_dir_if_not_exists(DIR)
    save_filename = DIR + time.strftime('%d-%m_%H:%M:%S') + '.txt'

    if run_type == 'met':
        # Metropolis sampling to find TOM params:
        accepted_history = ([1]+3*[0])*25  # Wiki recommends acceptance should be 23% (for a Gaussian dist!)
        start_time = time.time()
        for step_number in range(np.int(total_number_steps)):
            step_size = iterate_metropolis_sampling(params, mlp, expert_trajs, num_ep_to_use, epsilon_sd,
                            start_time, step_number, total_number_steps, accepted_history, step_size, save_filename)

    elif run_type == 'zeroth':
        # Optimise the params to fit the data:
        start_time = time.time()
        # For each gradient decent step, find the gradient and step:
        for step_number in range(np.int(total_number_steps)):
            find_gradient_and_step_multi_tom(params, mlp, expert_trajs, num_ep_to_use, lr, epsilon_sd,
                                            start_time, step_number, total_number_steps)
    elif run_type == 'acc':
        raise ValueError("Not done yet!")
        # Just find the top-1 and top-2 accuracy:

        # PERSON_PARAMS_TOMcheck = {"COMPLIANCE_TOMcheck": 0.8, "TEAMWORK_TOMcheck": 0.7,
        #     "RETAIN_GOALS_TOMcheck": 0.6, "WRONG_DECISIONS_TOMcheck": 0.1, "THINKING_PROB_TOMcheck": 0.5,
        #     "PATH_TEAMWORK_TOMcheck": 0.4, "RATIONALITY_COEFF_TOMcheck": 3, "PROB_PAUSING_TOMcheck": 0.2}
        # params["PERSON_PARAMS_TOMcheck"] = PERSON_PARAMS_TOMcheck
        tom_number = 'check'
        multi_tom_agent = ToMAgent(params, tom_number).get_multi_agent(mlp)
        start_time = time.time()
        top_1_acc, top_2_acc = find_top_12_accuracy(actions_from_data, expert_trajs, multi_tom_agent, num_ep_to_use,
                                                    number_states_with_acting)

        print('\nTop-1 accuracy: {}; Top-2 accuracy: {}; Finished acc calc in time {} secs'.format(
                                                                top_1_acc, top_2_acc, round(time.time() - start_time)))

    print('\nend')

