import rclpy
from rclpy.node import Node
from lrauv_msgs.msg import LRAUVInit
from std_msgs.msg import Header
import subprocess
import numpy as np
from .coordinates import EASTING_ORIGIN, NORTHING_ORIGIN, utm_to_lat_lon
from typing import List, Tuple

# TODO:
# find a way to visualize stuff o rviz
# check if the action controller works correctly
# define the random mechanism to spawn the vehicles around the area of lat 36.8 and -122
#
class LrauvSpawner(Node):

    def __init__(
            self,
            agent_depth:Tuple[float, float]=(0., 0.),
            landmark_depth:Tuple[float, float]=(5., 20.),
            min_distance:float=20., # min distance between vehicles
            max_distance:float=100., # maximum distance between vehicles
        ):
        super().__init__('lrauv_spawner')
        self.pub = self.create_publisher(LRAUVInit, '/lrauv/init', 10)
        self.entities_spawned = 0
        self.agent_depth = agent_depth
        self.landmark_depth = landmark_depth
        self.existing_positions = np.empty((0, 3)) # take trace of the positions of the spawned agents 
        self.min_distance = min_distance
        # internally, max_distance refers to maximum distance from map origin; dividing by 2 we ensure that any vehicle are not too far away than max_distance
        self.max_distance = max_distance / 2 

    def spawn(self, name:str='agent_1'):

        # assumes an agent is called agent_something, otherwise landmark
        spawn_agent = 'agent' in name

        lat, lon, z, heading = self.get_next_position(self.agent_depth if spawn_agent else self.landmark_depth)
        
        init_msg = LRAUVInit()
        init_msg.header = Header()
        init_msg.header.stamp = self.get_clock().now().to_msg()
        init_msg.id = name
        init_msg.init_lat = lat
        init_msg.init_lon = lon
        init_msg.init_z = z
        init_msg.init_pitch = 0.
        init_msg.init_roll = 0.
        init_msg.init_heading = heading
        init_msg.acomms_address = self.entities_spawned + 1

        self.pub.publish(init_msg)
        rclpy.spin_once(self, timeout_sec=0.0001)

        self.entities_spawned += 1

    def check_all_spawned(self, ids:List[str]):
        # check if all the ids in a list of ids are present in the output of 'gz topic -l'
        gz_topic_output = subprocess.check_output(['gz', 'topic', '-l']).decode('utf-8')
        while not all([i+'/state' in gz_topic_output for i in ids]):
            gz_topic_output = subprocess.check_output(['gz', 'topic', '-l']).decode('utf-8')

    def _generate_random_position(self, depth_range):
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(self.min_distance, self.max_distance)
        delta_easting = distance * np.cos(angle)
        delta_northing = distance * np.sin(angle)
        depth = np.random.uniform(depth_range[0], depth_range[1])
        return np.array([EASTING_ORIGIN + delta_easting, NORTHING_ORIGIN + delta_northing, depth])
    
    def _is_valid_position(self, new_position, existing_positions):
        if existing_positions.size == 0:
            return True
        distances = np.sqrt(np.sum((existing_positions - new_position) ** 2, axis=1))
        return np.all(distances >= self.min_distance)
    
    def get_next_position(self, depth_range):
        while True:
            new_position = self._generate_random_position(depth_range)
            if self._is_valid_position(new_position, self.existing_positions):
                self.existing_positions = np.vstack((self.existing_positions, new_position))
                lat, lon = utm_to_lat_lon(new_position[0], new_position[1])
                z = new_position[2]
                heading = np.random.uniform(0., 360.)
                return lat, lon, z, heading

    



    



