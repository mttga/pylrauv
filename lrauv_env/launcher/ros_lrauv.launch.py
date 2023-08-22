import os

from ament_index_python.packages import get_package_share_directory

import launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node

# helper function to create the rviz configuration dynamically
rviz_config_template = """
Panels:
  - Class: rviz_common/Displays
    Help Height: 78
    Name: Displays
    Property Tree Widget:
      Expanded:
        - /Global Options1
        - /Status1
      Splitter Ratio: 0.5
    Tree Height: 1085
  - Class: rviz_common/Selection
    Name: Selection
  - Class: rviz_common/Tool Properties
    Expanded:
      - /2D Goal Pose1
      - /Publish Point1
    Name: Tool Properties
    Splitter Ratio: 0.5886790156364441
  - Class: rviz_common/Views
    Expanded:
      - /Current View1
    Name: Views
    Splitter Ratio: 0.5
  - Class: rviz_common/Time
    Experimental: false
    Name: Time
    SyncMode: 0
    SyncSource: ""
Visualization Manager:
  Class: ""
  Displays:
    {agent_paths}
    {landmark_paths}
  Enabled: true
  Global Options:
    Background Color: 48; 48; 48
    Fixed Frame: map
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
      Hide Inactive Objects: true
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
    - Class: rviz_default_plugins/FocusCamera
    - Class: rviz_default_plugins/Measure
      Line color: 128; 128; 0
    - Class: rviz_default_plugins/SetInitialPose
      Covariance x: 0.25
      Covariance y: 0.25
      Covariance yaw: 0.06853891909122467
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /initialpose
    - Class: rviz_default_plugins/SetGoal
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /goal_pose
    - Class: rviz_default_plugins/PublishPoint
      Single click: true
      Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /clicked_point
  Transformation:
    Current:
      Class: rviz_default_plugins/TF
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 172.017822265625
      Enable Stereo Rendering:
        Stereo Eye Separation: 0.05999999865889549
        Stereo Focal Distance: 1
        Swap Stereo Eyes: false
        Value: false
      Focal Point:
        X: 0
        Y: 0
        Z: 0
      Focal Shape Fixed Size: true
      Focal Shape Size: 0.05000000074505806
      Invert Z Axis: false
      Name: Current View
      Near Clip Distance: 0.009999999776482582
      Pitch: 1.5697963237762451
      Target Frame: <Fixed Frame>
      Value: Orbit (rviz)
      Yaw: 1.3604005575180054
    Saved: ~
Window Geometry:
  Displays:
    collapsed: false
  Height: 1376
  Hide Left Dock: false
  Hide Right Dock: false
  QMainWindow State: 000000ff00000000fd000000040000000000000156000004c6fc0200000008fb0000001200530065006c0065006300740069006f006e00000001e10000009b0000005c00fffffffb0000001e0054006f006f006c002000500072006f007000650072007400690065007302000001ed000001df00000185000000a3fb000000120056006900650077007300200054006f006f02000001df000002110000018500000122fb000000200054006f006f006c002000500072006f0070006500720074006900650073003203000002880000011d000002210000017afb000000100044006900730070006c006100790073010000003b000004c6000000c700fffffffb0000002000730065006c0065006300740069006f006e00200062007500660066006500720200000138000000aa0000023a00000294fb00000014005700690064006500530074006500720065006f02000000e6000000d2000003ee0000030bfb0000000c004b0069006e0065006300740200000186000001060000030c00000261000000010000010f000004c6fc0200000003fb0000001e0054006f006f006c002000500072006f00700065007200740069006500730100000041000000780000000000000000fb0000000a00560069006500770073010000003b000004c6000000a000fffffffb0000001200530065006c0065006300740069006f006e010000025a000000b200000000000000000000000200000490000000a9fc0100000001fb0000000a00560069006500770073030000004e00000080000002e10000019700000003000009b80000003efc0100000002fb0000000800540069006d00650100000000000009b80000024400fffffffb0000000800540069006d0065010000000000000450000000000000000000000747000004c600000004000000040000000800000008fc0000000100000002000000010000000a0054006f006f006c00730100000000ffffffff0000000000000000
  Selection:
    collapsed: false
  Time:
    collapsed: false
  Tool Properties:
    collapsed: false
  Views:
    collapsed: false
  Width: 2488
  X: 72
  Y: 27
"""

def generate_paths_config(n_agents, n_landmarks, agent_path_template, landmark_path_template):
    agent_paths = ""
    landmark_paths = ""
    
    for i in range(1, n_agents + 1):
        agent_paths += agent_path_template.format(agent_id=i)
        
    for i in range(1, n_landmarks + 1):
        landmark_paths += landmark_path_template.format(landmark_id=i)
    
    return agent_paths, landmark_paths

def modify_rviz_config(n_agents, n_landmarks):
    agent_path_template = '''
    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: 92; 255; 92
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.5
      Name: Path
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 92; 255; 92
      Pose Style: None
      Radius: 0.5
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /agent_{agent_id}/path
      Value: true
    '''
    
    landmark_path_template = '''
    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: 0; 128; 255
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.5
      Name: Path
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 0; 128; 255
      Pose Style: None
      Radius: 0.5
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /landmark_{landmark_id}/path
      Value: true
    - Alpha: 1
      Buffer Length: 1
      Class: rviz_default_plugins/Path
      Color: 255; 90; 90
      Enabled: true
      Head Diameter: 0.30000001192092896
      Head Length: 0.20000000298023224
      Length: 0.30000001192092896
      Line Style: Lines
      Line Width: 0.5
      Name: Path
      Offset:
        X: 0
        Y: 0
        Z: 0
      Pose Color: 255; 90; 90
      Pose Style: None
      Radius: 0.5
      Shaft Diameter: 0.10000000149011612
      Shaft Length: 0.10000000149011612
      Topic:
        Depth: 5
        Durability Policy: Volatile
        Filter size: 10
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /landmark_{landmark_id}_tracking/path
      Value: true
    '''
    
    agent_paths, landmark_paths = generate_paths_config(n_agents, n_landmarks, agent_path_template, landmark_path_template)
    modified_rviz_config = rviz_config_template.format(agent_paths=agent_paths, landmark_paths=landmark_paths)

    path = '/tmp/rviz_config_temp.rviz'
    with open(path, 'w') as file:
        file.write(modified_rviz_config)

    return path

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
        # agents COMMUNICATION RECIEVING, Gazebo -> Ros
        f'/agent_{i}/rx#/{i}/rx@ros_gz_interfaces/msg/Dataframe[gz.msgs.Dataframe'
        for i in range(1, int(n_agents)+1)
    ]+[
        # lrauv init, ROS -> Gazebo 
        '/lrauv/init@lrauv_msgs/msg/LRAUVInit]lrauv_gazebo_plugins.msgs.LRAUVInit',
        # clock, Gazebo -> Ros 
        '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        # WORLD Control Service
        f'/world/{world_name}/control@ros_gz_interfaces/srv/ControlWorld',
        # Broker for sending message  ROS -> Gazebo
        f'/broker/msgs@ros_gz_interfaces/msg/Dataframe]gz.msgs.Dataframe',
        # ENTITY deletion (doesn't work properly)
        # f'/world/{world_name}/remove@ros_gz_interfaces/srv/DeleteEntity',
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
        config_path = modify_rviz_config(int(n_agents), int(n_landmarks))
        rviz = Node(
            package='rviz2',
            executable='rviz2',
            arguments=['-d', config_path],
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
