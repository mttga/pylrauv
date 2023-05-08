import rclpy
from controllers import LrauvEntityController, LrauvAgentController
from launcher import RosLrauvLauncher
from utils.spawner import LrauvSpawner
from utils.world_controller import WorldController

# TODO:
# implement a proper ACT method
# implement linear, static and linear with change of direction movement
# enable parallelization of environments

class LrauvEnv:

    def __init__(self, n_agents, n_landmarks, use_gui=False):

        self.n_agents      = n_agents
        self.n_landmarks   = n_landmarks
        self.agents_ids    = [f'agent_{i}' for i in range(1, n_agents+1)]
        self.landmarks_ids = [f'landmark_{i}' for i in range(1, n_landmarks+1)]
        self.entities_ids  = self.agents_ids + self.landmarks_ids
        self.use_gui = use_gui
        self.started = False


    def start(self):

        # launch simulator
        self.sim = RosLrauvLauncher(self.n_agents, self.n_landmarks, self.use_gui)
        self.sim.launch()

        # prepare nodes
        rclpy.init()
        self.nodes = []
        # Create the node to control the world (pause/unpause)
        # this node should be initalized first because it waits to recieve the initial simulation time, i.e. ensures simulation is ready
        self.world_controller = WorldController()
        self.nodes.append(self.world_controller)

        # spawn the entities
        self.spawner = LrauvSpawner()
        for name in self.entities_ids:
            self.spawner.spawn(name=name)
        self.nodes.append(self.spawner) # take trace of all the nodes
        self.spawner.check_all_spawned(self.entities_ids) # this only checks that exists at least one gazebo topic for each entity

        # prepare entity controllers
        self.entities = {}
        for i, name in enumerate(self.entities_ids):
            if 'agent' in name:
                self.entities[name] = LrauvAgentController(name=name, comm_adress=i+1, entities_names=self.entities_ids) 
            else:
                self.entities[name] = LrauvEntityController(name=name, comm_adress=i+1)
            self.nodes.append(self.entities[name])
        self.agents    = {name:self.entities[name] for name in self.agents_ids}
        self.landmarks = {name:self.entities[name] for name in self.landmarks_ids}

        # send the initial range requests and step initially for a couple of secs
        self.world_controller.step_world(step_time=1)
        for _, agent in self.agents.items():
            agent.send_range_requests()
        self.world_controller.step_world(step_time=1) 
        self.started = True
        obs = {i:c.get_obs() for i,c in self.agents.items()}   
        states = {i:c.get_state() for i,c in self.entities.items()}        

        return obs, states

    def reset(self):
        if self.started:
            self.close()
        return self.start()
    
    def step(self, actions=None, step_time:int=60, comm_time:int=2):
        
        for i, c in self.entities.items():
            c.send_command()

        # step the world but leave a final room for sending the communciactions between robots
        self.world_controller.step_world(step_time-comm_time)

        # send range requests
        for name, agent in self.agents.items():
            agent.send_range_requests()

        # step the world additionally to allow for range responses to get back
        self.world_controller.step_world(comm_time)

        obs = {i:c.get_obs() for i,c in self.agents.items()}   
        states = {i:c.get_state() for i,c in self.entities.items()}
        return obs, states

    def close(self):
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        self.sim.terminate()
    
    @property
    def sim_time(self):
        return self.world_controller.simulation_time.sec



