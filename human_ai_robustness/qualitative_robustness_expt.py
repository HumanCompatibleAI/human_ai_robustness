import time, copy, json
import numpy as np
from argparse import ArgumentParser
import matplotlib.pyplot as plt; plt.rcdefaults()

from overcooked_ai_py.utils import mean_and_std_err
from overcooked_ai_py.agents.agent import AgentPair, RandomAgent, StayAgent
from overcooked_ai_py.mdp.actions import Direction
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, PlayerState, ObjectState, OvercookedState
from overcooked_ai_py.planning.planners import MediumLevelPlanner
from human_aware_rl.ppo.ppo_pop import get_ppo_agent, make_tom_agent, get_ppo_run_seeds, PPO_DATA_DIR
from human_aware_rl.data_dir import DATA_DIR
from human_aware_rl.imitation.behavioural_cloning import get_bc_agent_from_saved
from human_aware_rl.utils import set_global_seed

from human_ai_robustness.agent import ToMModel
from human_ai_robustness.import_person_params import import_manual_tom_params


ALL_LAYOUTS = ["counter_circuit", "coordination_ring", "bottleneck", "room", "centre_objects", "centre_pots"]

no_counters_params = {
    'start_orientations': False,
    'wait_allowed': False,
    'counter_goals': [],
    'counter_drop': [],
    'counter_pickup': [],
    'same_motion_goals': True
}

def get_layout_horizon(layout, horizon_length):
    """Return the horizon for given layout/length of task"""
    # TODO: Clean this function: either make horizon a hardcoded property of each test, or turn this into a global dictionary
    # extra_time = 0 if test_agent.__class__ is ToMModel else 0  # For test runs, e.g. if we want to give the TOM extra time
    extra_time = 0
    if extra_time != 0:
        print('>>>>>>> Extra time = {} <<<<<<<<'.format(extra_time))
    if horizon_length == 'short':
        return extra_time + 10

    elif horizon_length == 'medium':
        if layout in ['coordination_ring', 'centre_pots']:
            return extra_time + 15
        else:
            return extra_time + 20

    elif horizon_length == 'long':
        if layout == 'counter_circuit':
            return extra_time + 30
        elif layout == 'coordination_ring':
            return extra_time + 25


# TODO: Should be using AgentEvaluator everywhere for standard mdp/mlp setups
def make_mdp(layout):
    # Make the standard mdp for this layout:
    mdp = OvercookedGridworld.from_layout_name(layout, start_order_list=['any'] * 100, cook_time=20, rew_shaping_params=None)
    return mdp

def make_mlp(mdp):
    no_counters_params['counter_drop'] = mdp.get_counter_locations()
    no_counters_params['counter_goals'] = mdp.get_counter_locations()
    return MediumLevelPlanner.from_pickle_or_compute(mdp, no_counters_params, force_compute=False)


##############################
# INITIAL STATES SETUP UTILS #
##############################

def make_ready_soup_at_loc(loc):
    return ObjectState('soup', loc, ('onion', 3, 20))

def make_soup_missing_one_onion(loc):
    return ObjectState('soup', loc, ('onion', 2, 0))


class InitialStatesCreator(object):

    def __init__(self, varying_params, constants, mdp):
        self.state_params = varying_params
        self.constants = constants
        self.mdp = mdp

    def get_initial_states(self, success_info=None, display=False):
        """
        With each initial state there will also be an optional success_info dict that can help with assessing the
        success of the test
        """
        states = []

        # TODO: This framework probably needs to be improved some more. Right now it's somewhat arbitrary that
        # the 3 elements of this tuple are H, R, and Objects. We should probably turn it into a dict and 
        # accept other types of thigns too, depending if it's ever used in the tests?
        for variation_params_dict in self.state_params[self.mdp.layout_name]:
            
            # Unpack info from variation params dict
            # TODO: if necessary for other tests to have different variation fields, we should make
            # the variation_params_dict have all possible data pieces and be either {data} or None
            # and then assert that every data piece is either in the variation_params_dict or in the self.constants
            # before proceeding
            # We could then overwrite all None variation_params_dict items (potentially this would require making them
            # lambdas too)
            h_loc = variation_params_dict["h_loc"]
            r_loc = variation_params_dict["r_loc"]

            # Players
            if "h_orientation_fn" in self.constants.keys():
                h_orientation = self.constants["h_orientation_fn"]()
            else: 
                h_orientation = Direction.random_direction()
            h_state = PlayerState(h_loc, h_orientation, held_object=self.constants["h_held"](h_loc))
            
            if "r_orientation_fn" in self.constants.keys():
                r_orientation = self.constants["r_orientation_fn"]()
            else: 
                r_orientation = Direction.random_direction()
            r_state = PlayerState(r_loc, r_orientation, held_object=self.constants["r_held"](r_loc))
            players = [h_state, r_state]

            # Objects
            objects = {}
            if "objects" in variation_params_dict.keys():
                objects = copy.deepcopy(self.constants["objects"])
                for obj_name, obj_loc_list in variation_params_dict["objects"].items():
                    for obj_loc in obj_loc_list:
                        objects[obj_loc] = self.custom_object_creation(obj_name, obj_loc)

            # Overcooked state
            # TODO: Should have all order lists be None, but seems to break things?
            s = OvercookedState(players, objects, order_list=['any'] * 100).deepcopy()
            states.append(s)

        return [(s, success_info) for s in states]

    def custom_object_creation(self, obj_name, obj_loc):
        """Allows for custom names (ready_soup, etc.)"""
        if obj_name in ObjectState.ALL_OBJECT_TYPES:
            return ObjectState(obj_name, obj_loc)
        elif obj_name == "ready_soup":
            return make_ready_soup_at_loc(obj_loc)
        else:
            raise ValueError("Unrecognized custom object type {}".format(obj_name))


###################
# TESTING CLASSES #
###################


class AbstractRobustnessTest(object):
    """
    Defines a specific robustness test
    
    # NOTE: For all tests, H_model is on index 0 and trained_agent is on index 1!

    # NOTE: a lot of the robustness tests could even be collapsed to testing the action probabilities for certain
    1-timestep-horizon situations (prob of picking up the soup > threshold). Won't be able to change this for neurips
    but might be something we want to do later. This would make tests easier to code and to interpret, as we don't have
    to think about interaction effects. Eventually might want to have both.
    """

    # Constant attributes
    ALL_TEST_TYPES = ["state_robustness", "agent_robustness", "memory"]

    # Attributes meant to be overwitten by subclasses
    valid_layouts = ALL_LAYOUTS
    test_types = None

    def __init__(self, mdp, testing_horizon, trained_agent, trained_agent_type, agent_run_name, num_rollouts_per_initial_state=1, print_info=False, display_runs=False):
        self.mdp = mdp
        self.layout = mdp.layout_name
        self.env_horizon = get_layout_horizon(self.layout, testing_horizon)
        self.num_rollouts_per_initial_state = num_rollouts_per_initial_state

        self.print_info = print_info
        self.display_runs = display_runs

        # Just a string of the name
        self.trained_agent_type = trained_agent_type
        self.agent_run_name = agent_run_name
        self.success_rate = self.evaluate_agent_on_layout(trained_agent)        
        self._check_valid_class()

    def to_dict(self):
        """To enable pickling if one wants to save the test data for later processing"""
        return {
            "layout": self.layout,
            "env_horizon": self.env_horizon,
            "test_types": self.test_types,
            "num_rollouts_per_initial_state": self.num_rollouts_per_initial_state,
            "trained_agent_type": self.trained_agent_type,
            "agent_run_name": self.agent_run_name,
            "success_rate": self.success_rate
        }

    def setup_human_model(self):
        raise NotImplementedError()

    def evaluate_agent_on_layout(self, trained_agent):
        H_model = self.setup_human_model()

        subtest_successes = []

        for (initial_state, success_info) in self.get_initial_states():

            for _ in range(self.num_rollouts_per_initial_state):
                # Check it's a valid state:
                self.mdp._check_valid_state(initial_state)

                # Setup env
                env = OvercookedEnv(self.mdp, start_state_fn=lambda: initial_state, horizon=self.env_horizon)

                # Play with the tom agent from this state and record score
                agent_pair = AgentPair(H_model, trained_agent)
                final_state = env.get_rollouts(agent_pair, num_games=1, final_state=True, display=self.display_runs, info=False)["ep_observations"][0][-1]
                
                if self.print_info:
                    env.state = initial_state
                    print('\nInitial state:\n{}'.format(env))
                    env.state = final_state
                    print('\nFinal state:\n{}'.format(env))

                success = self.is_success(initial_state, final_state, success_info)
                subtest_successes.append(success)

                if self.print_info:
                    print(sum(subtest_successes)/len(subtest_successes))
                    print('Subtest successes: {}'.format(subtest_successes))

        return sum(subtest_successes) / len(subtest_successes)

    def is_success(self, initial_state, final_state, success_info=None):
        raise NotImplementedError()

    def _check_valid_class(self):
        assert all(test_type in self.ALL_TEST_TYPES for test_type in self.test_types), "You need to set the self.test_types class attribute for this specific test class, and each test type must be among the following: {}".format(self.test_types)
        assert all(layout in ALL_LAYOUTS for layout in self.valid_layouts)

##########
# TEST 1 #
##########

class Test1(AbstractRobustnessTest):

    # NOTE: all subtests are state_robustness tests excpet for 1ai
    test_types = ["state_robustness"]


class Test1ai(Test1):
    """
    Pick up a dish from a counter: H blocks dispenser (in layouts with only one dispenser)
    
    Details:
    - 4 different settings for R's location and the location of the dishes
    - Both pots cooking
    - H holding onion facing South; R holding nothing
    - Success: R gets a dish or changes the pot state?

    POSSIBLE ADDITIONS: Give H no object. More positions for R.
    """

    valid_layouts = ['bottleneck', 'room', 'coordination_ring', 'counter_circuit']
    test_types = ["memory"]

    def get_initial_states(self):
        initial_states_params = {
            'counter_circuit': [
                {   "h_loc": (1, 2),     "r_loc": (1, 1),    "objects": { "dish": [(0, 1)]}                 },
                {   "h_loc": (1, 2),     "r_loc": (1, 1),    "objects": { "dish": [(0, 1), (1, 0), (6, 0)]} },
                {   "h_loc": (1, 2),     "r_loc": (6, 1),    "objects": { "dish": [(6, 0)],               } },
                {   "h_loc": (1, 2),     "r_loc": (6, 1),    "objects": { "dish": [(0, 1), (1, 0), (6, 0)]} }
            ],
            'coordination_ring': [
                {   "h_loc": (1, 2),     "r_loc": (2, 1),    "objects": { "dish": [(2, 0)],               } },
                {   "h_loc": (1, 2),     "r_loc": (2, 1),    "objects": { "dish": [(2, 0), (1, 0), (0, 1)]} },
                {   "h_loc": (1, 2),     "r_loc": (3, 3),    "objects": { "dish": [(4, 3)],               } },
                {   "h_loc": (1, 2),     "r_loc": (3, 3),    "objects": { "dish": [(4, 3), (4, 2), (3, 4)]} }
            ]
        }
        constants = {
            "h_held": lambda h_loc: ObjectState("onion", h_loc),
            "h_orientation_fn": lambda: Direction.SOUTH, # TODO: Do we actually want this? South will mean different things in different layouts anyways
            "r_held": lambda r_loc: None,
            "objects": { loc : make_ready_soup_at_loc(loc) for loc in self.mdp.get_pot_locations() }
        }
        return InitialStatesCreator(initial_states_params, constants, self.mdp).get_initial_states()

    def setup_human_model(self):
        return StayAgent()

    def is_success(self, initial_state, final_state, success_info=None):
        trained_agent = final_state.players[1]
        r_has_dish = trained_agent.has_object() and trained_agent.get_object().name == 'dish'
        # To change, soups must have either moved from the pot (picked up), delivered, or created (which is hard as all pots are already full)
        soups_have_changed = initial_state.all_objects_by_type['soup'] != final_state.all_objects_by_type['soup']
        success = r_has_dish or soups_have_changed
        if success and self.print_info:
            print('PPO has object, or the pot state has changed --> success!')
        return success


class Test1aii(Test1):
    """
    Pick up a dish from a counter: counter object is much closer than dispenser
    
    A: 
    - Counter object is onion
    - One pot is full, other one needs just one onion
    - Human is holding dish
    
    B:
    - Counter object is dish
    - All pots are full
    - Human is holding dish
    
    Success: R picks up the counter object
    """

    def get_initial_states(self):

        initial_states_params_A = {
            'coordination_ring': [
                {   "h_loc": (1, 3),     "r_loc": (2, 1),    "objects": { "onion": [(2, 2)]}                 },
                {   "h_loc": (1, 3),     "r_loc": (2, 1),    "objects": { "onion": [(2, 2), (2, 0), (0, 1)]} },
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "onion": [(2, 2)],               } },
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "onion": [(2, 2), (2, 0), (4, 2)]} }
            ]
        }

        pot_locations = self.mdp.get_pot_locations()
        first_pot_loc = pot_locations[0]
        other_pots_loc = pot_locations[1:]
        objects = { first_pot_loc : make_soup_missing_one_onion(first_pot_loc) }
        objects.update( { loc : make_ready_soup_at_loc(loc) for loc in other_pots_loc } )

        constants_A = {
            "h_held": lambda h_loc: ObjectState("dish", h_loc),
            "h_orientation_fn": lambda: Direction.random_direction(),
            "r_held": lambda r_loc: None,
            "r_orientation_fn": lambda: Direction.random_direction(),
            "objects": objects
        }
        variant_A_states = InitialStatesCreator(initial_states_params_A, constants_A, self.mdp).get_initial_states(success_info="onion")
        

        initial_states_params_B = {
            'coordination_ring': [
                {   "h_loc": (1, 3),     "r_loc": (2, 1),    "objects": { "dish": [(2, 2)]}                 },
                {   "h_loc": (1, 3),     "r_loc": (2, 1),    "objects": { "dish": [(2, 2), (2, 0), (0, 1)]} },
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "dish": [(2, 2)],               } },
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "dish": [(2, 2), (2, 0), (4, 2)]} }
            ]
        }

        constants_B = {
            "h_held": lambda h_loc: ObjectState("dish", h_loc),
            "h_orientation_fn": lambda: Direction.random_direction(),
            "r_held": lambda r_loc: None,
            "r_orientation_fn": lambda: Direction.random_direction(),
            "objects": { loc : make_ready_soup_at_loc(loc) for loc in self.mdp.get_pot_locations() }
        }
        variant_B_states = InitialStatesCreator(initial_states_params_B, constants_B, self.mdp).get_initial_states(success_info="dish")

        return variant_A_states + variant_B_states

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        trained_agent = final_state.players[1]

        object_type = success_info
        initial_counter_obj_locations = initial_state.unowned_objects_by_type[object_type]
        final_counter_objs_locations = final_state.unowned_objects_by_type[object_type]

        all_initial_objects_still_in_same_position = all(init_loc in final_counter_objs_locations for init_loc in initial_counter_obj_locations)

        # TODO: Should make this more stringent to make sure that it was the trained agent who picked up the object
        # Could make this more stringent by requiring trained agent to still be holding object or have successfully used it
        success = not all_initial_objects_still_in_same_position 
        if success and self.print_info:
            print('PPO interacted with one of the counter objects --> success!')
        return success
        

class Test1aiii(Test1):
    """
    Pick up a dish from a counter: Soup on the counter
    
    Details:
    - H holding onion
    - Pots empty
    - R close to soup(s)

    Success: R picking up any soup
    """

    def get_initial_states(self):
        initial_states_params = {
            'coordination_ring': [
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "ready_soup": [(2, 2)],               } },
                {   "h_loc": (1, 3),     "r_loc": (3, 2),    "objects": { "ready_soup": [(2, 2), (4, 2), (4, 3)]} },
                {   "h_loc": (1, 1),     "r_loc": (3, 3),    "objects": { "ready_soup": [(4, 3)],               } },
                {   "h_loc": (1, 1),     "r_loc": (3, 3),    "objects": { "ready_soup": [(3, 4), (4, 3), (4, 2)]} }
            ]
        }
        constants = {
            "h_held": lambda h_loc: ObjectState("onion", h_loc),
            "h_orientation_fn": lambda: Direction.random_direction(),
            "r_held": lambda r_loc: None,
            "r_orientation_fn": lambda: Direction.random_direction(),
            "objects": {}
        }
        return InitialStatesCreator(initial_states_params, constants, self.mdp).get_initial_states()

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        trained_agent = final_state.players[1]
        
        initial_counter_obj_locations = initial_state.unowned_objects_by_type["soup"]
        final_counter_objs_locations = final_state.unowned_objects_by_type["soup"]

        num_initial_objects_that_changed_position = sum(init_loc not in final_counter_objs_locations for init_loc in initial_counter_obj_locations)

        # TODO: Should make this more stringent to make sure that it was the trained agent who picked up the object!!!
        # This is actually an issue here. Harder to do without the events tracking code.
        # Could make this more stringent by requiring trained agent to still be holding object or have successfully used it
        success = num_initial_objects_that_changed_position >= 1

        # HACK: simply check that it's not the ToM agent that is holding the soup, make conditions for success more stringent
        tom_agent = final_state.players[0]
        tom_agent_holding_soup = False if not tom_agent.has_object() else tom_agent.get_object().name == "soup"
        if tom_agent_holding_soup:
            success = num_initial_objects_that_changed_position >= 2

        if success and self.print_info:
            print('PPO interacted with one of the counter objects --> success!')
        return success


class Test1bi(Test1):
    """
    Interacting with counters -> Drop objects onto counter -> R holding the wrong object

    Test: R is holding the wrong object, and must drop it
    Details:Two variants:
                A) R has D when O needed (both pots empty)
                B) R has O when two Ds needed (both pots cooked)
            For both A and B:
                Starting locations in STPs
                Other player (H) is the median TOM
                H has nothing
    """

    def get_initial_states(self):
        initial_states_params = {
            'coordination_ring': [
                # top-R / bottom-L:
                {   "h_loc": (3, 2),     "r_loc": (3, 1) },
                {   "h_loc": (2, 1),     "r_loc": (3, 1) },
                {   "h_loc": (1, 1),     "r_loc": (1, 3) },
                {   "h_loc": (3, 3),     "r_loc": (1, 3) },
                # Both near dish/soup:
                {   "h_loc": (3, 3),     "r_loc": (1, 2) },
                {   "h_loc": (1, 1),     "r_loc": (2, 3) },
                # Diagonal:
                {   "h_loc": (3, 3),     "r_loc": (1, 1) },
                {   "h_loc": (1, 1),     "r_loc": (3, 3) }
            ],
        }
        constants_variant_A = {
            "h_held": lambda h_loc: None,
            "h_orientation_fn": lambda: Direction.random_direction(),
            "r_held": lambda r_loc: ObjectState("dish", r_loc),
            "r_orientation_fn": lambda: Direction.random_direction(),
            "objects": {}
        }

        constants_variant_B = {
            "h_held": lambda h_loc: None,
            "h_orientation_fn": lambda: Direction.random_direction(),
            "r_held": lambda r_loc: ObjectState("onion", r_loc),
            "r_orientation_fn": lambda: Direction.random_direction(),
            "objects": { loc : make_ready_soup_at_loc(loc) for loc in self.mdp.get_pot_locations() }
        }

        variant_A_states = InitialStatesCreator(initial_states_params, constants_variant_A, self.mdp).get_initial_states()
        variant_B_states = InitialStatesCreator(initial_states_params, constants_variant_B, self.mdp).get_initial_states()
        return variant_A_states + variant_B_states

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        trained_agent_initial_state = initial_state.players[1]
        initial_object = trained_agent_initial_state.get_object().name
        trained_agent_final_state = final_state.players[1]
        # Agent must have gotten rid of initial object
        success = not (trained_agent_final_state.has_object() and trained_agent_final_state.get_object().name == initial_object)
        if success and self.print_info:
            print('PPO no longer has the {} --> success!'.format(initial_object))
        return success


class Test1bii(Test1):
    """
    Drop objects onto counter: R holding the same object as H, but H is closer to using it
    
    Details:

    A:
    - H & R holding onion
    - 1 soup missing 1 onion, all other pots full
    - H next to pot, R far

    B:
    - H & R holding dish
    - 1 soup ready, others empty
    - H next to pot, R far

    Success: R picking up any soup
    """

    # test_types # TODO: could make a variant that is dependent on "how much faster H is than R", making it "agent_robustness"

    def get_initial_states(self):
        initial_states_params = {
            'coordination_ring': [
                { "h_loc": (3, 1),     "r_loc": (1, 1) },
                { "h_loc": (3, 1),     "r_loc": (1, 3) },
                { "h_loc": (3, 1),     "r_loc": (3, 3) },
            ]
        }

        pot_locations = self.mdp.get_pot_locations()
        first_pot_loc = pot_locations[0]
        other_pots_loc = pot_locations[1:]
        objects = { first_pot_loc : make_soup_missing_one_onion(first_pot_loc) }
        objects.update( { loc : make_ready_soup_at_loc(loc) for loc in other_pots_loc } )

        constants_A = {
            "h_held": lambda h_loc: ObjectState("onion", h_loc),
            "r_held": lambda r_loc: ObjectState("onion", r_loc),
            "objects": objects
        }
        variant_A_states = InitialStatesCreator(initial_states_params, constants_A, self.mdp).get_initial_states(success_info="onion")
        

        constants_B = {
            "h_held": lambda h_loc: ObjectState("dish", h_loc),
            "r_held": lambda r_loc: ObjectState("dish", r_loc),
            "objects": { first_pot_loc : make_ready_soup_at_loc(first_pot_loc) }
        }
        variant_B_states = InitialStatesCreator(initial_states_params, constants_B, self.mdp).get_initial_states(success_info="dish")

        return variant_A_states + variant_B_states

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        trained_agent = final_state.players[1]
        
        inial_object_type = success_info

        # Success if R not holding the initial object
        success = not (trained_agent.has_object() and trained_agent.get_object().name == inial_object_type)
        if success and self.print_info:
            print('PPO no longer has the initial object ({}) --> success!'.format(inial_object_type))
        return success


##########
# TEST 2 #
##########


class Test2(AbstractRobustnessTest):
    
    # TODO: think more about this classification, and in general we should double check them all at the end
    test_types = ["state_robustness"]


class Test2a(Test2):
    """
    Getting out the way of H: R in the way of H, where H has the right object

    A:
    - H has onion, onion needed in pot

    B:
    - H has dish, dish needed

    Success: pot state has changed
    """

    def get_initial_states(self):
        initial_states_params_AB = {
            'coordination_ring': [
                { "h_loc": (1, 1),     "r_loc": (2, 1) },
                { "h_loc": (2, 1),     "r_loc": (3, 1) },
                { "h_loc": (3, 2),     "r_loc": (3, 1) },
                { "h_loc": (3, 3),     "r_loc": (3, 2) },
            ]
        }
        constants_A = {
            "h_held": lambda h_loc: ObjectState("onion", h_loc),
            "r_held": lambda r_loc: None,
            "objects": { loc : make_soup_missing_one_onion(loc) for loc in self.mdp.get_pot_locations() }
        }
        constants_B = {
            "h_held": lambda h_loc: ObjectState("dish", h_loc),
            "r_held": lambda r_loc: None,
            "objects": { loc : make_ready_soup_at_loc(loc) for loc in self.mdp.get_pot_locations() }
        }

        initial_states_A = InitialStatesCreator(initial_states_params_AB, constants_A, self.mdp).get_initial_states()
        initial_states_B = InitialStatesCreator(initial_states_params_AB, constants_B, self.mdp).get_initial_states()
        return initial_states_A + initial_states_B

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        initial_soup_state = initial_state.unowned_objects_by_type["soup"]
        final_soup_state = final_state.unowned_objects_by_type["soup"]
        success = initial_soup_state != final_soup_state
        if success and self.print_info:
            print('The pot state has changed --> success!')
        return success


class Test2b(Test2):
    """
    Getting out the way of H: H is holding a soup, and R is on the shortest path for H to deliver soup

    - H has soup

    Success: H not holding soup anymore
    """

    def get_initial_states(self):
        initial_states_params = {
            'coordination_ring': [
                { "h_loc": (1, 3),     "r_loc": (2, 3) },
                { "h_loc": (3, 1),     "r_loc": (3, 2) },
                { "h_loc": (3, 2),     "r_loc": (3, 3) }
            ]
        }
        constants = {
            "h_held": lambda h_loc: make_ready_soup_at_loc(h_loc),
            "r_held": lambda r_loc: None
        }
        return InitialStatesCreator(initial_states_params, constants, self.mdp).get_initial_states()

    def setup_human_model(self):
        return make_median_tom_agent(self.mdp)

    def is_success(self, initial_state, final_state, success_info=None):
        tom_agent = final_state.players[0]
        tom_holding_soup = tom_agent.has_object() and tom_agent.get_object().name == "soup"
        success = not tom_holding_soup
        if success and self.print_info:
            print('ToM not longer holding soup --> success!')
        return success


##########
# TEST 3 #
##########

class Test3(AbstractRobustnessTest):
    
    test_types = ["agent_robustness"]


##########
# TEST 4 #
##########

class Test4(AbstractRobustnessTest):
    
    test_types = ["agent_robustness", "memory"]


#####################
# AGENT SETUP UTILS #
#####################

def setup_agents_to_evaluate(mdp, agent_type, agent_run_name, agent_seeds, agent_save_location):
    assert agent_save_location == "local", "Currently anything else is unsupported"

    if agent_type != "ppo":
        assert agent_seeds is None, "For all agent types except ppo agents, agent_seeds should be None"

    if agent_type == "ppo":
        seeds = get_ppo_run_seeds(agent_run_name) if agent_seeds is None else agent_seeds
        agents = []
        for seed in seeds:
            ppo_agent_base_path = PPO_DATA_DIR + agent_run_name + "/"
            agent, _ = get_ppo_agent(ppo_agent_base_path, seed=seed)
            agents.append(agent)
    elif agent_type == "bc":
        raise [get_bc_agent(agent_run_name)]
    elif agent_type == "tom":
        agents = [make_median_tom_agent(mdp)]
    elif agent_type == "opt_tom":
        raise NotImplementedError("need to implement this")
    elif agent_type == "rnd":
        agents = [RandomAgent()]
    else:
        raise ValueError("Unrecognized agent type")

    assert len(agents) > 0
    return agents

def get_bc_agent(agent_run_name):
    """Return the BC agent for this layout and seed"""
    raise NotImplementedError("Should port over code from ppo_pop to ensure that the BC agent used is the same as the one used for PPO_BC_1 training")
    bc_agent, _ = get_bc_agent_from_saved(agent_run_name, unblock_if_stuck=True, stochastic=True, overwrite_bc_save_dir=None)
    return bc_agent


##################
# MAKE TOM UTILS #
##################

# TODO-Paul: check all these methods actually get the right models. The code behind this is too complicated for me to
# understand it in a short amount of time

def make_test_tom(mdp):
    # TODO: Not sure what type of agent this is supposed to be - OPT?
    mlp = make_mlp(mdp)
    test_agent = make_test_tom_agent(mdp.layout_name, mlp, tom_num=test_agent[3])
    print('Setting prob_pausing = 0')
    test_agent.prob_pausing = 0

def make_median_tom_agent(mdp):
    """Make the Median TOM agent -- with params such that is has the median score with other manual param TOMs"""
    mlp = make_mlp(mdp)
    # TODO: Not sure if this is right? Especially SELECT_TOM, only changed it becasue it was breaking.
    _, alternate_names_params, _ = import_manual_tom_params(mdp.layout_name, 1, MAXLIKE=True, SELECT_TOM="False")
    return ToMModel.from_alternate_names_params_dict(mlp, alternate_names_params[0])

def make_test_tom_agent(mdp, tom_num):
    # TODO: What is this?
    """Make a TOM from the VAL OR TRAIN? set used for ppo"""
    mlp = make_mlp(mdp)
    VAL_TOM_PARAMS, TRAIN_TOM_PARAMS, _ = import_manual_tom_params(mdp.layout_name, 20)
    tom_agent = make_tom_agent(mlp)
    tom_agent.set_tom_params(None, None, TRAIN_TOM_PARAMS, tom_params_choice=int(tom_num))
    return tom_agent


############################
# MAIN TEST RUN MANAGEMENT #
############################


all_tests = [Test1ai, Test1aii, Test1aiii, Test1bi, Test1bii, Test2a, Test2b]


def run_tests(tests_to_run, layout, num_avg, agent_type, agent_run_name, agent_save_location, agent_seeds, print_info, display_runs):

    # Make all randomness deterministic
    set_global_seed(0)

    # Set up agent to evaluate
    mdp = make_mdp(layout)
    agents_to_eval = setup_agents_to_evaluate(mdp, agent_type, agent_run_name, agent_seeds, agent_save_location)

    tests = {}
    for test_class in all_tests:
        results_across_seeds = []

        for agent_to_eval in agents_to_eval:
            # TODO: Fix env horizon -> currently I've just set it to be medium acrosss all tests
            test_object = test_class(mdp, "medium", trained_agent=agent_to_eval, trained_agent_type=agent_type, agent_run_name=agent_run_name, num_rollouts_per_initial_state=num_avg, print_info=print_info, display_runs=display_runs)
            results_across_seeds.append(test_object.to_dict())

        tests[test_object.__class__.__name__] = aggregate_test_results_across_seeds(results_across_seeds)

    # TODO: once we have these objects, we can easily apply filtering on all the data to generate
    # test-type specific plots and so on.

    print("Test results", tests)

    state_robustness_tests = filter_tests_by_attribute(tests, "test_types", ["state_robustness"])
    print(get_average_success_rate_across_tests(state_robustness_tests))

    return tests


###########################
# RESULT PROCESSING UTILS #
###########################

def aggregate_test_results_across_seeds(results):
    for result_dict in results:
        for k, v in result_dict.items():
            if k != "success_rate":
                # All dict entries across seeds should be the same except for the success rate
                assert v == results[0][k]

    final_dict = copy.deepcopy(results[0])
    del final_dict["success_rate"]
    final_dict["success_rate_mean_and_se"] = mean_and_std_err([result["success_rate"] for result in results])
    return final_dict

def filter_tests_by_attribute(tests_dict, attribute, value):
    """
    Returns tests that have `attribute` == `value`
    """
    filtered_tests = {}
    for test_name, test_data_dict in tests_dict.items():
        if test_data_dict[attribute] == value:
            filtered_tests[test_name] = test_data_dict
    return filtered_tests

def get_average_success_rate_across_tests(tests_dict):
    return np.mean([test["success_rate_mean_and_se"][0] for test in tests_dict.values()])


##########################
# COMMAND LINE INTERFACE #
##########################

if __name__ == "__main__":
    """
    Run a qualitative experiment to test robustness of a trained agent. This code works through a suite of tests,
    largely involving putting the test-subject-agent in a specific state, with a specific other player, then seeing if 
    they can still play Overcooked from that position.
    """
    parser = ArgumentParser()
    parser.add_argument("-t", "--tests_to_run", default="all")
    parser.add_argument("-l", "--layout", help="layout", required=True)
    parser.add_argument("-n", "--num_avg", type=int, required=False, default=1)
    parser.add_argument("-a_t", "--agent_type", type=str, required=True, default="ppo") # Must be one of ["ppo", "bc", "tom", "opt_tom"]
    parser.add_argument("-a_n", "--agent_run_name", type=str, required=False, help='e.g. lstm_expt_cc0')
    parser.add_argument("-a_s", "--agent_seeds", type=str, required=False, help='[9999, 8888]')
    parser.add_argument("-r", "--agent_save_location", required=False, type=str, help="e.g. server or local", default='local') # NOTE: removed support for this temporarily
    parser.add_argument("-pr", "--print_info", default=False, action='store_true')
    parser.add_argument("-dr", "--display_runs", default=False, action='store_true')

    args = parser.parse_args()
    run_tests(**args.__dict__)
