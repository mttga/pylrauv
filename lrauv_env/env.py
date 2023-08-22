import rclpy
from controllers import LrauvEntityController, LrauvAgentController, LrauvTeamController
from controllers.action import LinearController
from launcher.launcher import RosLrauvLauncher
from utils.spawner import LrauvSpawner
from utils.world_controller import WorldController
from utils.path_visualizer import LrauvPathVisualizer
from typing import Tuple

# TODO:
# implement a proper ACT method
# implement linear, static and linear with change of direction movement
# enable parallelization of environments

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
        landmark_action_controller:str='linear', # defines how the landmarks will move
        prop_range_agent:Tuple[float, float]=(25., 25.), # defines the speed range for agent
        prop_range_landmark:Tuple[float, float]=(0, 20.), # defines the speed range for landmark
        rudder_range_landmark:Tuple[float, float]=(0.10, 0.25), # defines the angle of movement change for landmarks
        tracking_method:str='ls' # the method used by the agents to track the landmarks, can be ls (Least Squares) or pf (Particle Filter)
    ):
        self.n_agents      = n_agents
        self.n_landmarks   = n_landmarks
        self.agents_ids    = [f'agent_{i}' for i in range(1, n_agents+1)]
        self.landmarks_ids = [f'landmark_{i}' for i in range(1, n_landmarks+1)]
        self.entities_ids  = self.agents_ids + self.landmarks_ids
        self.render = render
        self.started = False

        self.agent_depth = agent_depth
        self.landmark_depth = landmark_depth
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.prop_range_agent = prop_range_agent
        self.prop_range_landmark = prop_range_landmark
        self.rudder_range_landmark = rudder_range_landmark
        self.tracking_method = tracking_method

        # set the landmark controller
        if landmark_action_controller=='linear':
            self.landmark_controller = LinearController
        else:
            raise NotImplementedError(f"Action controller {landmark_action_controller} not implemented")

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
        for name in self.entities_ids:
            self.spawner.spawn(name=name)
        self.nodes.append(self.spawner) # take trace of all the nodes
        self.spawner.check_all_spawned(self.entities_ids) # this only checks that exists at least one gazebo topic for each entity

        # prepare entity controllers
        self.entities = {}
        for i, name in enumerate(self.entities_ids):
            if 'agent' in name:
                action_controller = self.landmark_controller(self.prop_range_agent)
                self.entities[name] = LrauvAgentController(
                    name=name,
                    comm_adress=i+1,
                    action_controller=action_controller,
                    entities_names=self.entities_ids,
                    method=self.tracking_method
                ) 
            else:
                action_controller = self.landmark_controller(self.prop_range_landmark, self.rudder_range_landmark)
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

        return obs, states

    def reset(self):
        if self.started:
            self.close()
        return self.start()
    
    def step(self, actions=None, step_time:int=60, comm_time:int=2):
        
        # send actions
        for e in self.entities.values():
            e.send_action(actions)

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
        return obs, states

    def close(self):
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        self.sim.terminate()
    
    @property
    def sim_time(self):
        return self.world_controller.simulation_time.sec



