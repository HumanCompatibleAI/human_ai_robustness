

from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
import copy
import numpy as np

if __name__ == "__main__":
    """Find the best BC (in terms of avg performance with all other BCs)"""

    # Variables to change
    layout_name = 'room'
    BC_SEEDS = [2222, 2599, 2732, 2897, 3264, 3468, 4373, 4859, 537, 5874, 6036, 6216, 6744, 6921, 705, 7768, 7891, 9225, 9845, 9893]
    BC_LOCAL_DIR = '/home/pmzpk/bc_runs/'
    NUM_GAMES = 3

    assert len(BC_SEEDS) == 20

    print('Layout: {}; Games: {}; Num seeds: {}'.format(layout_name, NUM_GAMES, len(BC_SEEDS)))

    cook_time = 20
    start_order_list = 100 * ['any']

    mdp = OvercookedGridworld.from_layout_name(layout_name, start_order_list=start_order_list,
                                               cook_time=cook_time, rew_shaping_params=None)
    env = OvercookedEnv(mdp, horizon=400)

    # Make all BCs:
    bc_agent_store = []
    for i in range(len(BC_SEEDS)):
        bc_name = layout_name + "_train_{}".format(BC_SEEDS[i])
        print("LOADING BC MODEL FROM: {}{}".format(BC_LOCAL_DIR, bc_name))
        bc_agent, bc_params = get_bc_agent_from_saved(bc_name, unblock_if_stuck=True, stochastic=True,
                                                      overwrite_bc_save_dir=BC_LOCAL_DIR)
        bc_agent.set_mdp(mdp)
        bc_agent_store.append(bc_agent)

    scores = []

    # Find score for each BC:
    for i in range(len(BC_SEEDS)):

        score_this_bc = 0
        bc_agent_player = bc_agent_store[i]

        # Play with each BC:
        for j in range(len(BC_SEEDS)):

            if i != j:
                bc_agent_opponent = bc_agent_store[j]
            else:
                bc_agent_opponent = copy.deepcopy(bc_agent_store[j])

            # Play with each index:
            for player_idx in range(2):

                print("Agent {} playing with agent {}, index {}".format(i, j, player_idx))

                # Probably not needed as get_rollouts resets??
                bc_agent_player.reset()
                bc_agent_opponent.reset()

                if player_idx == 0:
                    agent_pair = AgentPair(bc_agent_player, bc_agent_opponent)
                elif player_idx == 1:
                    agent_pair = AgentPair(bc_agent_opponent, bc_agent_player)

                trajs = env.get_rollouts(agent_pair, num_games=NUM_GAMES, final_state=False, display=False)
                sparse_rews = trajs["ep_returns"]
                avg_sparse_rew = np.mean(sparse_rews)

                print('Score this pair: {}'.format(avg_sparse_rew))

                score_this_bc += avg_sparse_rew

        avg_score_this_bc = score_this_bc / (len(BC_SEEDS)*2)  # *2 because 2 indices
        scores.append(avg_score_this_bc)
        print('\n\n\nAvg score this BC: {}\n\n\n'.format(avg_score_this_bc))

    winning_score = max(scores)
    winning_bc = np.argmax(scores)

    print("All scores: {}".format(scores))
    print("Layout: {}; Winning_score: {}; Winning_bc: {}; Winning bc's seed: {}".format(
                                                layout_name, winning_score, winning_bc, BC_SEEDS[winning_bc]))
