cmake_minimum_required(VERSION 3.8)
project(lrauv_msgs)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# set the message files
set(msg_files
  "msg/LRAUVInit.msg"
  "msg/LRAUVState.msg"
  "msg/LRAUVCommand.msg"
  "msg/LRAUVRangeBearingRequest.msg"
  "msg/LRAUVRangeBearingResponse.msg"
  "msg/LRAUVAcousticMessage.msg"
)

# generate the messages
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
