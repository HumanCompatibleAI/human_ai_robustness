
"""
Quick code to view an agent acting

"""

import pickle
import time
import unittest
import numpy as np
import random
import logging

from argparse import ArgumentParser
from overcooked_ai_py.agents.agent import Agent, AgentPair, FixedPlanAgent, CoupledPlanningAgent, StayAgent, \
    RandomAgent, GreedyHumanModel
from human_ai_robustness.agent import ToMModel, GreedyHumanModel_pk
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState, PlayerState, ObjectState
from overcooked_ai_py.mdp.actions import Direction, Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from overcooked_ai_py.agents.benchmarking import AgentEvaluator


def make_agent_pair(mlp):
    # Make agents:
    # perseverance0 = random.random()
    perseverance0 = 0.5
    # teamwork0 = random.random()
    teamwork0 = 0.5
    # retain_goals0 = random.random()
    retain_goals0 = 0.9
    # wrong_decisions0 = random.random()**30
    wrong_decisions0 = 0
    wrong_decisions1 = 0.5
    # thinking_prob0 = 1 - random.random()**30
    thinking_prob0 = 1
    # path_teamwork0 = 1 - random.random()**2
    path_teamwork0 = 0.5
    # rat_coeff0 = 2+random.random()*8
    rat_coeff0 = 3
    prob_pausing0 = 0
    a0 = ToMModel(mlp, player_index=0, perseverance=perseverance0, teamwork=teamwork0,
                  retain_goals=retain_goals0, wrong_decisions=wrong_decisions0,
                  thinking_prob=thinking_prob0, prob_pausing=prob_pausing0,
                  path_teamwork=path_teamwork0, rationality_coefficient=rat_coeff0)
    a1 = ToMModel(mlp, player_index=1, perseverance=perseverance0, teamwork=teamwork0,
                  retain_goals=retain_goals0, wrong_decisions=wrong_decisions1,
                  thinking_prob=thinking_prob0, prob_pausing=prob_pausing0,
                  path_teamwork=path_teamwork0, rationality_coefficient=rat_coeff0)
    a0.use_OLD_ml_action = False
    a1.use_OLD_ml_action = True
    # a0 = ToMModel(mlp, player_index=0, perseverance=0.9, teamwork=1, retain_goals=0.9,
    #                                  wrong_decisions=0.02, thinking_prob=1, path_teamwork=1, rationality_coefficient=2)
    # a1 = GreedyHumanModel_pk(mlp, player_index=0, perseverance=0.8)
    # a1 = RandomAgent(mlp)
    # print(perseverance0, teamwork0, retain_goals0, wrong_decisions0, thinking_prob0, prob_pausing0, path_teamwork0,
    #       rat_coeff0)
    # print('Player 0: teamwork: {:.1f}, retain: {:.1f}, wrong dec: {:.1f}'.format(teamwork0, retain_goals0, wrong_decisions0))
    return AgentPair(a0, a1)


if __name__ == "__main__" :
    """
    
    """
    parser = ArgumentParser()
    # parser.add_argument("-l", "--fixed_mdp", dest="layout",
    #                     help="name of the layout to be played as found in data/layouts",
    #                     required=True)
    parser.add_argument("-l", "--layout",
                        help="layout, (Choose from: sim, sc1, sch, uni, ran1, ran0)", required=True)
    print("\n****************************************\nNOTE: To watch play in debug, put breakpoint in "
          "overcooked_env.OvercookedEnv.run_agents, within loop 'while not done'\n*****************************************\n")
    args = parser.parse_args()
    layout = args.layout

    np.random.seed(41)

    # To print the agent's decisions:
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.ERROR)

    n, s = Direction.NORTH, Direction.SOUTH
    e, w = Direction.EAST, Direction.WEST
    stay, interact = Action.STAY, Action.INTERACT
    P, Obj = PlayerState, ObjectState

    DISPLAY = True
    start_order_list = ["any"]*20
    horizon = 2000
    explosion_time = 500
    r_shaping = 0
    cook_time = 5

    no_counters_params = {
        'start_orientations': False,
        'wait_allowed': False,
        'counter_goals': [],
        'counter_drop': [],
        'counter_pickup': [],
        'same_motion_goals': True
    }

    if layout == 'uni':
        start_state = OvercookedState([P((2, 2), n), P((5, 2), n)], {}, order_list=start_order_list)
        mdp = OvercookedGridworld.from_layout_name('asymmetric_advantages', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        mdp_params = {"layout_name": "asymmetric_advantages", "start_order_list": start_order_list, "cook_time": cook_time}
    # elif layout == 'sc1':
    #     start_state = OvercookedState([P((1, 2), n), P((1, 1), n)], {}, order_list=start_order_list)
    #     mdp = OvercookedGridworld.from_layout_name('scenario1_s', start_order_list=start_order_list,
    #                                                cook_time=cook_time, rew_shaping_params=None)
    #     mdp_params = {"layout_name": "scenario1_s", "start_order_list": start_order_list, "cook_time": cook_time}
    elif layout == 'ran1':
        start_state = OvercookedState([P((1, 2), n), P((1, 1), n)], {}, order_list=start_order_list)
        mdp = OvercookedGridworld.from_layout_name('coordination_ring', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        mdp_params = {"layout_name": "coordination_ring", "start_order_list": start_order_list, "cook_time": cook_time}
    elif layout == 'ran0':
        start_state = OvercookedState([P((1, 2), n), P((3, 2), n)], {}, order_list=start_order_list)
        mdp = OvercookedGridworld.from_layout_name('forced_coordination', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        mdp_params = {"layout_name": "forced_coordination", "start_order_list": start_order_list, "cook_time": cook_time}
    elif layout == 'ran3':
        start_state = OvercookedState([P((1, 2), n), P((6, 2), n)], {}, order_list=start_order_list)
        mdp = OvercookedGridworld.from_layout_name('counter_circuit', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        mdp_params = {"layout_name": "counter_circuit", "start_order_list": start_order_list, "cook_time": cook_time}
    elif layout == 'sim':
        start_state = OvercookedState([P((2, 2), n), P((2, 1), n)], {}, order_list=start_order_list)
        mdp = OvercookedGridworld.from_layout_name('cramped_room', start_order_list=start_order_list,
                                                   cook_time=cook_time, rew_shaping_params=None)
        mdp_params = {"layout_name": "cramped_room", "start_order_list": start_order_list, "cook_time": cook_time}
    else:
        raise ValueError('layout not recognised')

    # Doing this means that all counter locations are allowed to have objects dropped on them AND be "goals" (I think!)
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()

    mlp = MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)

    # Added since restructuring changes:

    env_params = {"start_state_fn": lambda: start_state, "horizon": horizon}
    mlp_params = no_counters_params
    # one_counter_params = { 'start_orientations': False, 'wait_allowed': False, 'counter_goals': valid_counters,
    #     'counter_drop': valid_counters, 'counter_pickup': [], 'same_motion_goals': True }
    # mlp_params=one_counter_params

    # Make and evaluate agents:
    ap = make_agent_pair(mlp)
    a_eval = AgentEvaluator(mdp_params=mdp_params, env_params=env_params, mlp_params=mlp_params)
    a_eval.evaluate_agent_pair(ap, display=True)


