import numpy as np
from typing import Tuple

# static variables
PROP_MIN = -31.4
PROP_MAX = 31.4
RUDDER_MIN = -0.261799
RUDDER_MAX = 0.261799

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
        self.steps_until_next_rudder_change = np.random.randint(2, 9)

    def sample_action(self):
        self.step_counter += 1

        if self.step_counter == self.steps_until_next_rudder_change:
            rudder_angle = np.random.uniform(*self.rudder_angle_range)
            self.steps_until_next_rudder_change = self.step_counter + np.random.randint(2, 9)
        else:
            rudder_angle = 0.

        action = (self.propulser_velocity, rudder_angle)
        self.previous_actions.append(action)
        return action