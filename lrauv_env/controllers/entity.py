import rclpy
from rclpy.node import Node
from lrauv_msgs.msg import LRAUVState, LRAUVCommand
from std_msgs.msg import Header
import random

class LrauvEntityController(Node):

    # static variables
    prop_min = -31.4
    prop_max = 31.4
    rudder_min = -0.261799
    rudder_max = 0.261799

    def __init__(
        self,
        name:str='agent_1',
        comm_adress:int=1,
    ):
        super().__init__(f'lrauv_{name}_controller')
        self.is_agent = 'agent' in name
        self.name = name
        # state subscriber
        self.state = None
        self.state_sub = self.create_subscription(
            LRAUVState,
            f"{name}/state_topic",
            self._state_callback,
            10
        )
        # controller publisher
        self.command_pub = self.create_publisher(LRAUVCommand, f"{name}/command_topic", 10)
        # communication adress
        self.comm_adress = comm_adress

    def _state_callback(self, msg):
        self.state = {
            'x':msg.pos.x,
            'y':msg.pos.y,
            'z':msg.pos.z,
            'vel_x':msg.rate_uvw.x,
            'vel_y':msg.rate_uvw.x,
            'vel_z':msg.rate_uvw.x,
        }

    def get_state(self):
        while self.state is None:
            rclpy.spin_once(self, timeout_sec=0.00001)
        state = self.state
        self.state = None
        return state
    
    def send_command(self, prop_action=None, rudder_action=None):
        if prop_action is None and rudder_action is None:
            prop_action, rudder_action = self.get_random_action()
        msg = LRAUVCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.prop_omega_action = prop_action
        msg.rudder_angle_action = rudder_action
        self.command_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.0001)

    def get_random_action(self):
        prop_action = random.uniform(0., 20.)
        rudder_action = random.uniform(-0.1, 0.1)
        return prop_action, rudder_action
