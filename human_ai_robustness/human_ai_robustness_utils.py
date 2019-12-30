from overcooked_ai_py.agents.agent import AgentPair, ToMModel
import numpy as np


def eval_and_viz_tom(additional_params, env, model,run_info):
    """
    The ppo agent will play with a selection of other agents, to evaluate performance, including TOMs with
    params 0 and 1 (who they'll also train with), and two different BC agents. They all play with both indices.
    """

    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
    from overcooked_ai_py.mdp.layout_generator import LayoutGenerator
    from overcooked_ai_py.planning.planners import NO_COUNTERS_PARAMS, MediumLevelPlanner
    from human_aware_rl.baselines_utils import get_agent_from_model

    display_eval_games = additional_params["DISPLAY_EVAL_GAMES"]

    mdp_params = additional_params["mdp_params"]
    mdp_gen_params = additional_params["mdp_generation_params"]
    mdp_fn = LayoutGenerator.mdp_gen_fn_from_dict(mdp_params=mdp_params, **mdp_gen_params)
    overcooked_env = OvercookedEnv(mdp=mdp_fn, **additional_params["env_params"])
    mlp = MediumLevelPlanner.from_pickle_or_compute(overcooked_env.mdp, NO_COUNTERS_PARAMS, force_compute=True)

    ppo_agent = get_agent_from_model(model, additional_params["sim_threads"], is_joint_action=False)
    ppo_agent.set_mdp(overcooked_env.mdp)

    if not additional_params["LOCAL_TESTING"]:  # Only evaluate with all 8 agents when not doing local testing

        # Loop over both indices and 2 different TOM params
        for ppo_index in range(2):
            for tom_number in range(2):

                tom_index = 1 - ppo_index
                print('\nPPO index {} | Playing with TOM{}\n'.format(ppo_index, tom_number))
                tom_agent = make_tom_model(env, mlp, tom_number, tom_index)

                if ppo_index == 0:
                    agent_pair = AgentPair(ppo_agent, tom_agent)
                elif ppo_index == 1:
                    agent_pair = AgentPair(tom_agent, ppo_agent)

                trajs = overcooked_env.get_rollouts(agent_pair, num_games=additional_params["NUM_EVAL_GAMES"],
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)

                run_info["rew_ppo_idx{}_tom{}".format(ppo_index, tom_number)].append(avg_sparse_rew)

                # To observe play:
                if display_eval_games:
                    overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

            for bc_number in range(2):

                print('\nPPO index {} | Playing with BC{}\n'.format(ppo_index, bc_number))

                bc_agent = env.bc_agent0 if bc_number == 0 else env.bc_agent1

                if ppo_index == 0:
                    agent_pair = AgentPair(ppo_agent, bc_agent)
                elif ppo_index == 1:
                    agent_pair = AgentPair(bc_agent, ppo_agent)

                trajs = overcooked_env.get_rollouts(agent_pair, num_games=additional_params["NUM_EVAL_GAMES"],
                                                    final_state=False, display=False)  # reward shaping not needed
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)

                run_info["rew_ppo_idx{}_bc{}".format(ppo_index, bc_number)].append(avg_sparse_rew)

                # To observe play:
                if display_eval_games:
                    overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

    elif additional_params["LOCAL_TESTING"]:

        ppo_index = 0
        tom_index = 1 - ppo_index
        tom_number = 0

        print('\nPPO index {} | Playing with TOM{}\n'.format(ppo_index, tom_number))
        tom_agent = make_tom_model(env, mlp, tom_number, tom_index)

        if ppo_index == 0:
            agent_pair = AgentPair(ppo_agent, tom_agent)
        elif ppo_index == 1:
            agent_pair = AgentPair(tom_agent, ppo_agent)

        trajs = overcooked_env.get_rollouts(agent_pair, num_games=additional_params["NUM_EVAL_GAMES"],
                                            final_state=False, display=False)  # reward shaping not needed
        sparse_rews = trajs["ep_returns"]
        avg_sparse_rew = np.mean(sparse_rews)

        run_info["rew_ppo_idx{}_tom{}".format(ppo_index, tom_number)].append(avg_sparse_rew)

        # To observe play:
        if display_eval_games:
            overcooked_env.get_rollouts(agent_pair, num_games=1, final_state=False, display=True)

    return run_info


def make_tom_model(env, mlp, hm_number, agent_index):

    perseverance = env.tom_params[hm_number]["PERSEVERANCE_HM{}".format(hm_number)]
    teamwork = env.tom_params[hm_number]["TEAMWORK_HM{}".format(hm_number)]
    retain_goals = env.tom_params[hm_number]["RETAIN_GOALS_HM{}".format(hm_number)]
    wrong_decisions = env.tom_params[hm_number]["WRONG_DECISIONS_HM{}".format(hm_number)]
    thinking_prob = env.tom_params[hm_number]["THINKING_PROB_HM{}".format(hm_number)]
    path_teamwork = env.tom_params[hm_number]["PATH_TEAMWORK_HM{}".format(hm_number)]
    rationality_coeff = env.tom_params[hm_number]["RATIONALITY_COEFF_HM{}".format(hm_number)]
    prob_pausing = env.tom_params[hm_number]["PROB_PAUSING_HM{}".format(hm_number)]

    return ToMModel(mlp=mlp, player_index=agent_index, perseverance=perseverance,
                    teamwork=teamwork, retain_goals=retain_goals,
                    wrong_decisions=wrong_decisions, thinking_prob=thinking_prob,
                    path_teamwork=path_teamwork, rationality_coefficient=rationality_coeff,
                    prob_pausing=prob_pausing)
