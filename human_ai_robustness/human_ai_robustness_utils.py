#
# def convert_layout_names_if_required(layout_name):
#     """If the layout name uses the NEW naming, then change it to the OLD naming"""
#     if layout_name == "cramped_room":
#         return "simple"
#     elif layout_name == "coordination_ring":
#         return "random1"
#     elif layout_name == "forced_coordination":
#         return "random0"
#     elif layout_name == "counter_circuit":
#         return "random3"
#     elif layout_name == "asymmetric_advantages":
#         return "unident_s"
#     else:
#         return layout_name

#TODO: We don't need this: can just use value=LinearAnnealer then take 1-value!
class LinearAnnealerZeroToOne():
    """Anneals from 0 to 1 (it reaches 1 when timestep == horizon)"""

    def __init__(self, horizon):
        self.horizon = horizon

    def param_value(self, timestep):
        curr_value = min(timestep / self.horizon, 1)
        assert 0 <= curr_value <= 1
        return curr_value