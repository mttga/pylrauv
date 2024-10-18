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
        rudder_angle_range:Tuple[float]=(RUDDER_MIN, RUDDER_MAX)
    ):
        self.propulser_range = propulser_range
        self.rudder_angle_range = rudder_angle_range
        self.propulser_velocity = np.random.uniform(*propulser_range)
        self.step_counter = 0
        self.previous_actions = []
        #self.steps_until_next_rudder_change = np.random.randint(5, 20)
        self.steps_until_next_rudder_change = np.random.randint(2, 5) # to collect more data

    def sample_action(self):
    
        if self.step_counter == self.steps_until_next_rudder_change:
            rudder_angle = np.random.uniform(*self.rudder_angle_range) * np.random.choice([-1,1])
            self.steps_until_next_rudder_change = self.step_counter + np.random.randint(2, 5)
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
    

class ConstantVelocityDiscreteRudderController:
    def __init__(
        self,
        propulser_range:Tuple[float]=(0., PROP_MAX/2),
        discrete_action_mapping:List[float]=DISCRETE_ACTION_MAPPING
    ):
        self.propulser_range = propulser_range
        self.discrete_action_mapping = discrete_action_mapping
        self.propulser_velocity = np.random.uniform(*propulser_range)
        self.previous_actions = []

    def process_action(self, action):

        self.previous_actions.append(action)
        return (self.propulser_velocity, self.discrete_action_mapping[action])

"""
class LinearController:
    def __init__(
        self,
        propulser_range:Tuple[float]=(0., PROP_MAX/2),
        rudder_angle_range:Tuple[float]=(RUDDER_MIN, RUDDER_MAX)
    ):
        self.propulser_range = propulser_range
        self.rudder_angle = rudder_angle_range[1]
        self.propulser_velocity = propulser_range[1]
        self.step_counter = 0

    def sample_action(self):
    
        if self.step_counter%5==0:
            rudder_angle = self.rudder_angle
        else:
            rudder_angle = 0.
        
        action = (self.propulser_velocity, rudder_angle)
        self.step_counter += 1
        return action
"""