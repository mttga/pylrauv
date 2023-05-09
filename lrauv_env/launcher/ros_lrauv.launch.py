import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition

from launch_ros.actions import Node


def launch_setup(context, *args, **kwargs):

    world_name = 'empty_environment'
    nodes = []
    server_mode = bool(int(launch.substitutions.LaunchConfiguration('server_mode').perform(context)))
    current_dir = os.path.dirname(os.path.abspath(__file__))
    gz_args = os.path.join(current_dir, 'worlds', world_name+'.sdf')
    if server_mode:
        gz_args = '-s ' + gz_args
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': gz_args
        }.items(),
    )
    nodes.append(gz_sim)

    # Bridge topic
    n_agents = launch.substitutions.LaunchConfiguration('n_agents').perform(context)
    n_landmarks = launch.substitutions.LaunchConfiguration('n_landmarks').perform(context)
    arguments = [
        # agents STATE, Gazebo -> Ros
        f'/agent_{i}/state_topic@lrauv_msgs/msg/LRAUVState[lrauv_gazebo_plugins.msgs.LRAUVState'
        for i in range(1, int(n_agents)+1)
    ]+[
        # landmarks STATE, Gazebo -> Ros
        f'/landmark_{i}/state_topic@lrauv_msgs/msg/LRAUVState[lrauv_gazebo_plugins.msgs.LRAUVState'
        for i in range(1, int(n_landmarks)+1)
    ]+[
        # agents COMMMAND, Ros -> Gazebo
        f'/agent_{i}/command_topic@lrauv_msgs/msg/LRAUVCommand]lrauv_gazebo_plugins.msgs.LRAUVCommand'
        for i in range(1, int(n_agents)+1)
    ]+[
        # landmarks COMMAND, Ros -> Gazebo
        f'/landmark_{i}/command_topic@lrauv_msgs/msg/LRAUVCommand]lrauv_gazebo_plugins.msgs.LRAUVCommand'
        for i in range(1, int(n_landmarks)+1)
    ]+[
        # agents RANGE REQUEST, Ros -> Gazebo
        f'/agent_{i}/range_bearing/requests@lrauv_msgs/msg/LRAUVRangeBearingRequest]lrauv_gazebo_plugins.msgs.LRAUVRangeBearingRequest'
        for i in range(1, int(n_agents)+1)
    ]+[
        # landmarks RANGE REQUEST, Ros -> Gazebo
        f'/landmark_{i}/range_bearing/requests@lrauv_msgs/msg/LRAUVRangeBearingRequest]lrauv_gazebo_plugins.msgs.LRAUVRangeBearingRequest'
        for i in range(1, int(n_landmarks)+1)
    ]+[
        # agents RANGE RESPONSE, Ros -> Gazebo
        f'/agent_{i}/range_bearing/responses@lrauv_msgs/msg/LRAUVRangeBearingResponse[lrauv_gazebo_plugins.msgs.LRAUVRangeBearingResponse'
        for i in range(1, int(n_agents)+1)
    ]+[
        # landmarks RANGE RESPONSE, Ros -> Gazebo
        f'/landmark_{i}/range_bearing/responses@lrauv_msgs/msg/LRAUVRangeBearingResponse[lrauv_gazebo_plugins.msgs.LRAUVRangeBearingResponse'
        for i in range(1, int(n_landmarks)+1)
    ]+[
        # lrauv init, ROS -> Gazebo 
        '/lrauv/init@lrauv_msgs/msg/LRAUVInit]lrauv_gazebo_plugins.msgs.LRAUVInit',
        # clock, Gazebo -> Ros 
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # WORLD Control Service
        f'/world/{world_name}/control@ros_gz_interfaces/srv/ControlWorld',
        # ENTITY deletion (doesn't work properly)
        # f'/world/{world_name}/remove@ros_gz_interfaces/srv/DeleteEntity',
    ]+[
        # landmarks RANGE RESPONSE, Ros -> Gazebo
        f'/landmark_{i}/range_bearing/responses@lrauv_msgs/msg/LRAUVRangeBearingResponse[lrauv_gazebo_plugins.msgs.LRAUVRangeBearingResponse'
        for i in range(1, int(n_landmarks)+1)
    ]
    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=arguments,
        output='screen'
    )
    nodes.append(bridge)

    # RViz
    if not server_mode:
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            parameters=[
                {'use_sim_time': True},
            ]
        )
        nodes.append(rviz)

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('n_agents', default_value='2',
                              description='Number of agents that will perform tracking.'),
        DeclareLaunchArgument('n_landmarks', default_value='2',
                              description='Number of landmarks to be tracked.'),
        DeclareLaunchArgument('server_mode', default_value='0',
                              description='If true, does not display gz gui'),
        OpaqueFunction(function = launch_setup)
    ])
