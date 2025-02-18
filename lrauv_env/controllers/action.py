import numpy as np
from typing import Tuple, List

# static variables
PROP_MIN = -65.
PROP_MAX = 65.4
RUDDER_MIN = 0.10
RUDDER_MAX = 0.26

DISCRETE_ACTION_MAPPING = np.array([-0.24, -0.12, 0, 0.12, 0.24])

class LinearController:
    def __init__(
        self,
        propulser_range:Tuple[float]=(0., PROP_MAX/2),
        rudder_angle_range:Tuple[float]=(RUDDER_MIN, RUDDER_MAX),
        dirchange_time_range:Tuple[int]=(2, 5)
    ):
        self.propulser_range = propulser_range
        self.rudder_angle_range = rudder_angle_range
        self.dirchange_time_range = dirchange_time_range
        self.propulser_velocity = np.random.uniform(*propulser_range)
        self.step_counter = 0
        self.previous_actions = []
        self.steps_until_next_rudder_change = np.random.randint(*self.dirchange_time_range) # to collect more data

    def sample_action(self):
    
        if self.step_counter == self.steps_until_next_rudder_change:
            rudder_angle = np.random.uniform(*self.rudder_angle_range) * np.random.choice([-1,1])
            self.steps_until_next_rudder_change = self.step_counter + np.random.randint(*self.dirchange_time_range)
        else:
            rudder_angle = 0.

        action = (self.propulser_velocity, rudder_angle)
        self.previous_actions.append(action)
        self.step_counter += 1
        return action
    

class ConstantVelocityRudderController:
    def __init__(
        self,
        propulser_range:Tuple[float]=(0., PROP_MAX/2),
        rudder_angle_range:Tuple[float]=(RUDDER_MIN, RUDDER_MAX)
    ):
        self.propulser_range = propulser_range
        self.rudder_angle_range = rudder_angle_range
        self.propulser_velocity = np.random.uniform(*propulser_range)
        self.previous_actions = []

    def process_action(self, action):

        self.previous_actions.append(action)
        action = np.clip(action, self.rudder_angle_range[0], self.rudder_angle_range[1])
        return (self.propulser_velocity, action)
    
    def get_available_actions(self):
        raise NotImplementedError("Method not implemented for this controller.")
    

class ConstantVelocityDiscreteRudderController:
    def __init__(
        self,
        propulser_range:Tuple[float]=(0., PROP_MAX/2),
        rudder_angle_range:Tuple[float]=(RUDDER_MIN, RUDDER_MAX),
        discrete_action_mapping:List[float]=DISCRETE_ACTION_MAPPING
    ):
        self.propulser_range = propulser_range
        self.discrete_action_mapping = discrete_action_mapping
        self.propulser_velocity = np.random.uniform(*propulser_range)
        self.previous_actions = []

    def process_action(self, action):

        self.previous_actions.append(action)
        return (self.propulser_velocity, self.discrete_action_mapping[action])

    def get_available_actions(self):
        
        last_action = self.previous_actions[-1] if self.previous_actions else 0

        avail_actions = np.zeros(len(self.discrete_action_mapping))

        if last_action == 0:
            avail_actions[0] = 1
            avail_actions[1] = 1

        elif last_action == len(self.discrete_action_mapping)-1:
            avail_actions[-1] = 1
            avail_actions[-2] = 1

        else:
            avail_actions[last_action] = 1
            avail_actions[last_action-1] = 1
            avail_actions[last_action+1] = 1

        return avail_actions