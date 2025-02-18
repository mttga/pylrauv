# facilitates the mangament of multiple agents (expecially communication)

import numpy as np
from .agent import LrauvAgentController
from utils.world_controller import WorldController
from typing import List, Dict
import logging
logger = logging.getLogger(__name__) 

class LrauvTeamController:

    def __init__(self, agents:Dict[str, LrauvAgentController], world_controller:WorldController):

        self.agents = agents
        self.world_controller = world_controller
        

    def send_range_requests(self, comm_time=1):
        for _, agent in self.agents.items():
            agent.send_range_requests()
        self.world_controller.step_world(step_time=comm_time)
        self.range_request_sent = True


    def communicate(self, comm_time=1):
        # this should always be called after the range request is sent
        assert self.range_request_sent
        for _, agent in self.agents.items():
            agent.communicate()
        self.world_controller.step_world(step_time=comm_time)
        self.range_request_sent = False

    def get_obs(self, dt=30):
        return {name:agent.get_obs(dt=dt) for name, agent in self.agents.items()}

    def get_team_tracking(self, state, obs):
        """returns the best team's estimation of landmarks position (estimation of the closer landmarks)"""

        agents_names = [name for name in state.keys() if 'agent' in name]
        landmarks_names = [name for name in state.keys() if 'landmark' in name]

        # find the closer agent to each landmark
        ranges = np.array([[o[f'{landmark}_range'] for o in obs.values()] for landmark in landmarks_names])
        closer_agents = np.argmin(np.where(ranges==0, np.inf, ranges),axis=-1) # avoid 0 ranges (communication error)

        # get the x,y tracking for each landmark of the closer agent to that landmark
        team_tracking = {}
        for i, landmark in enumerate(landmarks_names):
            team_tracking[f'{landmark}_tracking_x'] = obs[agents_names[closer_agents[i]]][f'{landmark}_tracking_x']
            team_tracking[f'{landmark}_tracking_y'] = obs[agents_names[closer_agents[i]]][f'{landmark}_tracking_y']
            team_tracking[f'{landmark}_tracking_z'] = state[f'{landmark}']['z'] # for now z is assumed to be known

        return team_tracking


        
            