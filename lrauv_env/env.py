"""
LRAUV Environment for Underwater Tracking.

Mainly used for testing RL agents in LRAUV gazebo simulator. 

TODO:
- reward function
- info dictionary
"""

import rclpy
from controllers import LrauvEntityController, LrauvAgentController, LrauvTeamController
from controllers.action import LinearController, ConstantVelocityRudderController, ConstantVelocityDiscreteRudderController
from launcher.launcher import RosLrauvLauncher
from utils.spawner import LrauvSpawner
from utils.world_controller import WorldController
from utils.path_visualizer import LrauvPathVisualizer
from typing import Tuple

class LrauvEnv:

    def __init__(
        self,
        n_agents:int,
        n_landmarks:int,
        render:bool=False,
        agent_depth:Tuple[float, float]=(0., 0.), # defines the range of depth for spawning agents
        landmark_depth:Tuple[float, float]=(5., 20.), # defines the range of depth for spawinng landmarks
        min_distance:float=20., # min initial distance between vehicles
        max_distance:float=100., # maximum initial distance between vehicles
        landmark_controller:str='linear_random', # defines how the landmarks will move
        prop_range_agent:Tuple[float, float]=(30., 30.), # defines the speed range for agent
        landmark_rel_speed:Tuple[float, float]=(0., 0.5), # defines the speed range for landmark relative to agent
        rudder_range_landmark:Tuple[float, float]=(0.10, 0.24), # defines the angle of movement change for landmarks
        dirchange_time_range_landmark:Tuple[int, int]=(2, 10), # defines the time range for changing the direction of the landmarks
        tracking_method:str='ls', # the method used by the agents to track the landmarks, can be ls (Least Squares) or pf (Particle Filter)
        difficulty: str='medium', # the difficulty of the task, can be easy, medium, hard orexpert
        agent_controller:str='rudder_discrete', # defines how the agents will move
        depth_known:bool=True, # if the depth of the landmarks is known
        **tracking_args, # arguments for the tracking method
    ):

        assert difficulty in [
            "manual",
            "easy",
            "medium",
            "hard",
            "expert",
        ], "difficulty must be manual, easy, medium, hard or expert"

        if difficulty != 'manual':
            if difficulty == 'easy':
                landmark_rel_speed = (0., 0.35)
                dirchange_time_range_landmark=(5, 15)
            elif difficulty == 'medium':
                landmark_rel_speed = (0.15, 0.5)
                dirchange_time_range_landmark=(2, 10)
            elif difficulty == 'hard':
                landmark_rel_speed = (0.5, 0.7)
                dirchange_time_range_landmark=(5, 15)
            elif difficulty == 'expert':
                landmark_rel_speed = (0.83, 0.86)
                dirchange_time_range_landmark=(5, 15)

        self.n_agents      = n_agents
        self.n_landmarks   = n_landmarks
        self.agents_ids    = [f'agent_{i}' for i in range(1, n_agents+1)]
        self.landmarks_ids = [f'landmark_{i}' for i in range(1, n_landmarks+1)]
        self.entities_ids  = self.agents_ids + self.landmarks_ids
        self.render = render
        self.started = False

        # set the propulsor ranges for the landmark in relation to the agent
        prop_range_landmark = (
            landmark_rel_speed[0] * prop_range_agent[0],
            landmark_rel_speed[1] * prop_range_agent[1]
        )

        self.agent_depth = agent_depth
        self.landmark_depth = landmark_depth
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.prop_range_agent = prop_range_agent
        self.prop_range_landmark = prop_range_landmark
        self.rudder_range_landmark = rudder_range_landmark
        self.tracking_method = tracking_method
        self.depth_known = depth_known
        self.tracking_args = tracking_args

        if agent_controller=='linear_random':
            self.agent_controller = LinearController
        elif agent_controller=='rudder':
            self.agent_controller = ConstantVelocityRudderController
        elif agent_controller=='rudder_discrete':
            self.agent_controller = ConstantVelocityDiscreteRudderController
        else:
            raise NotImplementedError(f"Agent controller {agent_controller} not implemented.")

        # set the landmark controller
        self.landmark_controller_kwargs = {'rudder_angle_range': rudder_range_landmark}
        if landmark_controller=='linear_random':
            self.landmark_controller = LinearController
            self.landmark_controller_kwargs['dirchange_time_range'] = dirchange_time_range_landmark
        elif landmark_controller=='rudder':
            self.landmark_controller = ConstantVelocityRudderController
        elif landmark_controller=='rudder_discrete':
            self.landmark_controller = ConstantVelocityDiscreteRudderController
        else:
            raise NotImplementedError(f"Landmark controller {landmark_controller} not implemented")

    def start(self):

        # launch simulator
        self.sim = RosLrauvLauncher(self.n_agents, self.n_landmarks, self.render)
        self.sim.launch()

        # prepare nodes
        rclpy.init()
        self.nodes = []
        # Create the node to control the world (pause/unpause)
        # this node should be initalized first because it waits to recieve the initial simulation time, i.e. ensures simulation is ready
        self.world_controller = WorldController()
        self.nodes.append(self.world_controller)

        # spawn the entities
        self.spawner = LrauvSpawner(
            agent_depth=self.agent_depth,
            landmark_depth=self.landmark_depth,
            min_distance=self.min_distance,
            max_distance=self.max_distance
        )
        landmarks_depth = []
        for name in self.entities_ids:
            lat, lon, z, heading = self.spawner.spawn(name=name)
            if 'landmark' in name:
                landmarks_depth.append(z)
        
        self.nodes.append(self.spawner) # take trace of all the nodes
        self.spawner.check_all_spawned(self.entities_ids) # this only checks that exists at least one gazebo topic for each entity

        # prepare entity controllers
        self.entities = {}
        for i, name in enumerate(self.entities_ids):
            if 'agent' in name:
                action_controller = self.agent_controller(self.prop_range_agent)
                self.entities[name] = LrauvAgentController(
                    name=name,
                    comm_adress=i+1,
                    action_controller=action_controller,
                    entities_names=self.entities_ids,
                    method=self.tracking_method,
                    landmarks_depth=landmarks_depth if self.depth_known else None,
                    **self.tracking_args
                ) 
            else:
                action_controller = self.landmark_controller(self.prop_range_landmark, **self.landmark_controller_kwargs)
                self.entities[name] = LrauvEntityController(name=name, comm_adress=i+1, action_controller=action_controller)
            self.nodes.append(self.entities[name])
        self.agents    = {name:self.entities[name] for name in self.agents_ids}
        self.landmarks = {name:self.entities[name] for name in self.landmarks_ids}

        # use the team controller to facilitate the menagement of the agents
        self.team = LrauvTeamController(self.agents, self.world_controller)

        # add also the path visualizaer if render mode
        if self.render:
            self.visualizer = LrauvPathVisualizer(self.entities_ids)
            self.nodes.append(self.visualizer)

        # start for initial communication
        self.world_controller.step_world(step_time=1)
        self.started = True

        # these functions should be always called in this order: range_request -> get_state -> communicate -> get_obs
        self.team.send_range_requests()
        states = {i:c.get_state() for i,c in self.entities.items()}
        self.team.communicate()
        obs = self.team.get_obs()

        # get the current tracking estimation
        tracking = self.team.get_team_tracking(states, obs)

        if self.render:
            self.visualizer.update(tracking)     

        # update the states with the tracking information
        for l in self.landmarks_ids:
            states[l]['tracking_x'] = tracking[f'{l}_tracking_x']
            states[l]['tracking_y'] = tracking[f'{l}_tracking_y']
            states[l]['tracking_z'] = tracking[f'{l}_tracking_z']   

        return obs, states

    def reset(self):
        if self.started:
            self.close()
        return self.start()
    
    def step(self, actions=None, step_time:int=30, comm_time:int=2):
        
        # send actions
        for e in self.entities.values():
            if actions is None or e.name not in actions:
                action = None # None action will sample with the default controller
            else:
                action = actions[e.name]
            e.send_action(action)

        # step the world but leave a final room for sending the communciactions between robots
        self.world_controller.step_world(step_time-comm_time)

        # comunicate and get the state and observations
        self.team.send_range_requests(comm_time/2)
        states = {i:c.get_state() for i,c in self.entities.items()}
        self.team.communicate(comm_time/2)
        obs = self.team.get_obs(dt=step_time)

        # get the current tracking estimation
        tracking = self.team.get_team_tracking(states, obs)

        # spin the visualizer if render mode
        if self.render:
            self.visualizer.update(tracking)

        # update the states with the tracking information
        for l in self.landmarks_ids:
            states[l]['tracking_x'] = tracking[f'{l}_tracking_x']
            states[l]['tracking_y'] = tracking[f'{l}_tracking_y']
            states[l]['tracking_z'] = tracking[f'{l}_tracking_z']
        
        return obs, states
    
    def get_available_actions(self):
        return {name:c.action_controller.get_available_actions() for name, c in self.agents.items()}

    def close(self):
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        self.sim.terminate()
    
    @property
    def sim_time(self):
        return self.world_controller.simulation_time.sec



