import rclpy
from rclpy.node import Node
from lrauv_msgs.msg import LRAUVState, LRAUVCommand
from std_msgs.msg import Header
from .action import LinearController
from typing import Union

class LrauvEntityController(Node):

    def __init__(
        self,
        name:str='agent_1',
        comm_adress:int=1,
        action_controller:Union[LinearController,None]=None
    ):
        super().__init__(f'lrauv_{name}_controller')
        self.is_agent = 'agent' in name
        self.name = name
        # state subscriber
        self.state = None
        self.state_sub = self.create_subscription(
            LRAUVState,
            f"/{name}/state_topic",
            self._state_callback,
            10
        )
        # controller publisher
        self.command_pub = self.create_publisher(LRAUVCommand, f"/{name}/command_topic", 10)
        # communication adress
        self.comm_adress = comm_adress
        # action controller
        self.action_controller = action_controller
        if action_controller is None:
            self.action_controller = LinearController() # linear controller by default

    def _state_callback(self, msg):
        self.state = {
            'x':msg.pos.x,
            'y':msg.pos.y,
            'z':msg.pos.z,
            'vel_x':msg.pos_dot.x,
            'vel_y':msg.pos_dot.y,
            'vel_z':msg.pos_dot.z,
            'rph_x':msg.pos_rph.x,
            'rph_y':msg.pos_rph.y,
            'pqr_x':msg.rate_pqr.x,
            'pqr_y':msg.rate_pqr.y,
            'rud_ang':msg.rudder_angle,
            'prop_vel':msg.prop_omega,
        }

    def get_state(self):
        while self.state is None:
            rclpy.spin_once(self, timeout_sec=0.00001)
        state = self.state
        self.state = None
        return state
    
    def send_action(self, action=None):
        if action is None:
            prop_action, rudder_action = self.action_controller.sample_action()
        else:
            prop_action, rudder_action = self.action_controller.process_action(action)
        self._send_command(prop_action, rudder_action)
    
    def _send_command(self, prop_action:float, rudder_action:float):
        msg = LRAUVCommand()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.prop_omega_action = prop_action
        msg.rudder_angle_action = rudder_action
        msg.buoyancy_action = 0.0005 if 'landmark' in self.name else 0.
        msg.drop_weight = True if 'landmark' in self.name else False
        self.command_pub.publish(msg)
        rclpy.spin_once(self, timeout_sec=0.0001)
