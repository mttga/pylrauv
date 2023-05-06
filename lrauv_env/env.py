import rclpy
from controllers import LrauvAgentController
from launcher import RosLrauvLauncher
from utils.spawner import LrauvSpawner
from utils.world_controller import WorldController

# TODO:
# implement range bearing requests and responses
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
            self.spawner.spawn(name)
        self.nodes.append(self.spawner) # take trace of all the nodes
        self.spawner.check_all_spawned(self.entities_ids) # this only checks that exists at least one gazebo topic for each entity

        # prepare controllers
        self.controllers = {}
        for name in self.entities_ids:
            self.controllers[name] = LrauvAgentController(name)
            self.nodes.append(self.controllers[name])

        self.world_controller.step_world(step_time=1) # step initially for one sec
        states = {i:c.get_state() for i,c in self.controllers.items()}
        
        self.started = True
        return states

    def reset(self):
        if self.started:
            self.close()
        return self.start()
    
    def step(self, actions=None, step_time:int=30):
        
        for i, c in self.controllers.items():
            c.send_command()

        self.world_controller.step_world(step_time)

        states = {i:c.get_state() for i,c in self.controllers.items()}
        return states

    def close(self):
        for node in self.nodes:
            node.destroy_node()
        rclpy.shutdown()
        self.sim.terminate()
    
    @property
    def sim_time(self):
        return self.world_controller.simulation_time.sec



