import numpy as np

def get_hh_states_start_state_fn(layout_name, data_type):
    from human_aware_rl.human.process_dataframes import get_human_human_trajectories
    if data_type == "train" or data_type is True:
        hh_trajs_for_layout = get_human_human_trajectories([layout_name], "train")[layout_name]
    elif data_type == "test":
        hh_trajs_for_layout = get_human_human_trajectories([layout_name], "test")[layout_name]
    else:
        raise ValueError()

    hh_starts = np.concatenate(hh_trajs_for_layout["ep_observations"])

    # Need to set the order list for the TOMs to work with these hh_starts:
    for i in range(len(hh_starts)):
        hh_starts[i].order_list = ["any"] * 100

    start_state_fn = lambda: np.random.choice(hh_starts)
    return start_state_fn