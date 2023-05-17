import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Header, ColorRGBA
from geometry_msgs.msg import PoseStamped
from lrauv_msgs.msg import LRAUVState
import math
from functools import partial
from typing import List

class LrauvPathVisualizer(Node):

    agent_color    = ColorRGBA(r=0.0, g=1.0, b=0.0, a=1.0) 
    landmark_color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
    frame_id = 'map'

    def __init__(self, entities_ids:List[str]):
        super().__init__('path_visualizer')

        self.entities_ids = entities_ids
        self.subscribers = {}
        self.path_publishers = {}
        self.marker_publisher = self.create_publisher(MarkerArray, 'visualization_marker_array', 10)

        for name in self.entities_ids:
            self.subscribers[name] = self.create_subscription(LRAUVState, f"/{name}/state_topic", partial(self.update_callback, name=name), 10)
            self.path_publishers[name] = self.create_publisher(Path, f'/{name}/path', 10)

        self.paths  = {name: Path(header=Header(frame_id=self.frame_id)) for name in self.entities_ids}
        self.colors = {name:(self.agent_color if 'agent' in name else self.landmark_color) for name in self.entities_ids}
        self.updated = {name: False for name in self.entities_ids}
        self.legend_markers = []

        self.create_legend()

    def create_legend(self):

        for i, name in enumerate(self.entities_ids):
            color = self.colors[name]
            marker = self.create_legend_marker(name, i, color)
            self.legend_markers.append(marker)

        marker_array = MarkerArray(markers=self.legend_markers)
        self.marker_publisher.publish(marker_array)

    def create_legend_marker(self, name, idx, color):
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = name
        marker.id = idx
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = 1.0
        marker.pose.position.y = -0.5 * idx
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.5
        marker.color = color
        marker.text = name.capitalize()
        marker.lifetime = rclpy.duration.Duration(seconds=0).to_msg()
        return marker
        

    def update_callback(self, msg, name:str='agent_1'):
        path = self.paths[name]
        path.header.stamp = self.get_clock().now().to_msg()
        pose_stamped = self.create_pose_stamped(msg)
        path.poses.append(pose_stamped)
        self.path_publishers[name].publish(path)
        self.updated[name] = True

    def create_pose_stamped(self, msg):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = msg.header.stamp
        pose_stamped.header.frame_id = self.frame_id

        pose_stamped.pose.position.x = msg.pos.x
        pose_stamped.pose.position.y = msg.pos.y
        pose_stamped.pose.position.z = - msg.pos.z

        roll, pitch, yaw = msg.pos_rph.x, msg.pos_rph.y, msg.pos_rph.z
        q = Quaternion()
        q = self.euler_to_quaternion(roll, pitch, yaw)
        pose_stamped.pose.orientation = q

        return pose_stamped

    def euler_to_quaternion(self, roll, pitch, yaw):
        qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
        qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
        qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
        qw = math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) + math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)

        return Quaternion(x=qx, y=qy, z=qz, w=qw)

    def update(self):
        while rclpy.ok():
            # Spin the node until all paths have been updated
            if all(self.updated[name] for name in self.entities_ids):
                break
            rclpy.spin_once(self)

        # re-init the paths
        self.updated = {name: False for name in self.entities_ids}