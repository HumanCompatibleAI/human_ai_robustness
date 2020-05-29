import numpy as np
import pandas as pd
from human_aware_rl.data_dir import DATA_DIR

def get_trajs_for_new_data_format(data_path, layouts):
    """This loads the data then selects the data for the chosen layout, and returns joint_expert_trajs"""
    assert len(layouts) == 1, "Assuming only one layout is selected"
    main_trials = pd.read_pickle(DATA_DIR + data_path)
    joint_expert_trajs = main_trials[layouts[0]]
    # Rename so that labels are consistent:
    joint_expert_trajs['ep_observations'] = joint_expert_trajs.pop('ep_states')
    return joint_expert_trajs

def get_human_human_trajectories_new_data(layouts, dataset_type):
    """The new layouts have a slightly different format. This is a hacky solution to loading the new layouts' data
    #TODO: Fix this properly"""
    data_path = "human/anonymized/clean_{}_trials_new_layouts.pkl".format(dataset_type)
    joint_expert_trajs = get_trajs_for_new_data_format(data_path, layouts)
    return joint_expert_trajs  # Note that this returns joint expert trajs. This is find for data starts.

def get_hh_states_start_state_fn(layout_name, data_type):
    #TODO: This has become VERY hacky and needs to be sorted out
    from human_aware_rl.human.process_dataframes import get_human_human_trajectories
    if data_type == "train" or data_type is True:
        if layout_name in ['bottleneck', 'large_room', 'centre_objects', 'centre_pots']:
            hh_trajs_for_layout = get_human_human_trajectories_new_data([layout_name], "train")
        else:
            hh_trajs_for_layout = get_human_human_trajectories([layout_name], "train")[layout_name]
    elif data_type == "test":
        if layout_name in ['bottleneck', 'large_room', 'centre_objects', 'centre_pots']:
            hh_trajs_for_layout = get_human_human_trajectories_new_data([layout_name], "test")
        else:
            hh_trajs_for_layout = get_human_human_trajectories([layout_name], "test")[layout_name]
    else:
        raise ValueError()

    hh_starts = np.concatenate(hh_trajs_for_layout["ep_observations"])

    # Need to set the order list for the TOMs to work with these hh_starts:
    for i in range(len(hh_starts)):
        hh_starts[i].order_list = ["any"] * 100

    start_state_fn = lambda: np.random.choice(hh_starts)
    return start_state_fn