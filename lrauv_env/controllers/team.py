# facilitates the mangament of multiple agents (expecially communication)

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

    def get_obs(self, comm_time=4):
        return {name:agent.get_obs() for name, agent in self.agents.items()}



        
            