import rclpy
from rclpy.node import Node
from ros_gz_interfaces.srv import ControlWorld
from rosgraph_msgs.msg import Clock
from builtin_interfaces.msg import Time

class WorldController(Node):

    def __init__(self, world_topic='/world/empty_environment/control'):
        super().__init__(f'world_controller')

        # create the client for the world controlling service
        self.world_client = self.create_client(ControlWorld, world_topic)
        while not self.world_client.wait_for_service(timeout_sec=0.5):
            self.get_logger().info('world_control service not available, waiting again...')

        # clock subscriber
        self.simulation_time = None
        self.clock_sub = self.create_subscription(
            Clock,
            '/clock',
            self.clock_callback,
            10
        )

        # wait for recieving at least the initiali simulation time
        while self.simulation_time is None:
            rclpy.spin_once(self, timeout_sec=0.005)

    def step_world(self, step_time:int=60):

        # Ensure the simulation is paused
        request = ControlWorld.Request()
        request.world_control.pause = True
        self.world_control_request(request)

        # Run the simulation to the target time and then pause
        request.world_control.pause = False
        request.world_control.run_to_sim_time = self.get_target_time(step_time)
        self.world_control_request(request)
        # wait to reach the target time
        while self.simulation_time.sec < request.world_control.run_to_sim_time.sec:
            rclpy.spin_once(self, timeout_sec=0.005)

    def get_target_time(self, step_time:int=60):
        # creates the simulation time target in the future given a time interval to step
        target_time = Time()
        target_time.sec = self.simulation_time.sec + int(step_time)
        target_time.nanosec = self.simulation_time.nanosec + int((step_time % 1) * 1e9)
        if target_time.nanosec >= 1e9:
            target_time.sec += 1
            target_time.nanosec -= int(1e9)
        return target_time

    def world_control_request(self, request:ControlWorld):
        # Send the control request and wait for the response
        future = self.world_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if not response.success:
            self.get_logger().error('Failed to control the world')

    def clock_callback(self, msg):
        self.simulation_time = msg.clock