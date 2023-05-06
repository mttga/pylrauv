import rclpy
from rclpy.node import Node
from lrauv_msgs.msg import LRAUVInit
from std_msgs.msg import Header
import random
import subprocess
from typing import List

# TODO:
# define the random mechanism to spawn the vehicles around the area of lat 36.8 and -122
#
class LrauvSpawner(Node):

    def __init__(self, landmark_depth=10, remove_topic='/world/empty_environment/remove'):
        super().__init__('lrauv_spawner')
        self.pub = self.create_publisher(LRAUVInit, '/lrauv/init', 10)
        self.positions = {}
        self.agents_spawned = 0
        self.landmarks_spawned = 0
        self.entities_spawned = 0
        self.landmark_depth = 10.


    def spawn(self, name:str='agent_1'):

        # assumes an agent is called agent_something, otherwise landmark
        spawn_agent = 'agent' in name

        if spawn_agent:
            pos, heading = self.get_next_agent_position()
        else:
            pos, heading = self.get_next_landmark_position()
        
        init_msg = LRAUVInit()
        init_msg.header = Header()
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.id = name
        init_msg.init_lat = pos[0]
        init_msg.init_lon = pos[1]
        init_msg.init_z = pos[2]
        init_msg.init_pitch = 0.
        init_msg.init_roll = 0.
        init_msg.init_heading = heading
        init_msg.acomms_address = self.entities_spawned + 1

        self.pub.publish(init_msg)

        rclpy.spin_once(self, timeout_sec=0.001)

        if spawn_agent:
            self.agents_spawned += 1
        else:
            self.landmarks_spawned += 1
        self.entities_spawned += 1

    def check_all_spawned(self, ids:List[str]):
        
        # check if all the ids in a list of ids are present in the output of 'gz topic -l'
        gz_topic_output = subprocess.check_output(['gz', 'topic', '-l']).decode('utf-8')
        while not all([i+'/state' in gz_topic_output for i in ids]):
            gz_topic_output = subprocess.check_output(['gz', 'topic', '-l']).decode('utf-8')
    
    # UTM -> lat, lon
    def get_next_agent_position(self):
        lat = 36.7999992370605 + self.agents_spawned / 10000
        lon = -122.7 + self.agents_spawned / 10000
        z = 0.
        heading = random.uniform(0., 360.)
        return (lat, lon, z), heading

    def get_next_landmark_position(self):
        lat = 36.7999992370605 + self.landmarks_spawned / 10000
        lon = -122.7 + self.landmarks_spawned / 10000
        z = self.landmark_depth
        heading = random.uniform(0., 360.)
        return (lat, lon, z), heading
    



    



