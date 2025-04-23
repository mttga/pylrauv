# PYLRAUV

Python interface for controlling the [Gazebo LRAUV simulator](https://github.com/osrf/lrauv) using `rclpy`.


## Controllers

This repository provides several controllers to interact with the simulator:

- General control of the Gazebo simulator (e.g., start, step, clock): `utils.world_controller.WorldController`  
- Spawning LRAUV vehicles: `utils.spawner.LrauvSpawner`  
- Controlling general LRAUV vehicles: `controllers.LrauvAgentController`  
- Controlling LRAUV agents that communicate via acoustic signals and perform acoustic tracking: `controllers.LrauvAgentController`  
- Coordinating communication between a team of vehicles: `controllers.LrauvTeamController`

Check out `lrauv_env/env.py` to see how these controllers can be used to simulate various scenarios.


## Underwater Acoustic Tracking

We provide a complete environment for testing Underwater Acoustic Tracking (UAT) with LRAUV vehicles. In this setup, a team of LRAUV agents uses acoustic signals to estimate the positions of other vehicles. Each agent is equipped with its own tracking method (either Least Squares or Particle Filter) and shares observations with the rest of the team.

You can initialize the environment as follows:

```python
env = LrauvEnv(
    n_agents=2,                 # Number of tracking agents
    n_landmarks=2,              # Number of targets to track
    render=True,                # Enables Gazebo and RViz rendering
    prop_range_agent=(30.0, 30.0),
    difficulty='medium',        # One of: 'easy', 'medium', 'hard', 'expert'
    max_distance=150,           # Max initial distance between agents and landmarks
    tracking_method="pf",       # Use 'pf' for particle filter (efficient JAX implementation)
    agent_controller="rudder_discrete",  # Agents control their rudder via discrete actions
)
```



## Testing Agents from JaxMARL

We recommend using the accelerated distillation version of PyLRAUV provided by [JaxMarl](https://github.com/FLAIROx/JaxMARL/tree/utracking/jaxmarl/environments) (make sure atm you're using the ```utracking``` branch). To test agents trained in JaxMARL, use the following script:

```bash
python3 lrauv_env/test_jax_agent.py \
    --mode test \                         # Use 'test' to render or 'collect' to save trajectories efficiently
    --model_path models/mappo_transformer_hard.safetensors \  # Path to your trained model weights
    --num_agents 2 \
    --num_landmarks 2 \
    --difficulty medium \
    --episodes 10 \                       # Number of episodes to test
    --steps 100 \                         # Number of steps per episode
    --output_path outputs                 # Directory to store output logs
```


## Installation

Clone the repository and build the Docker image using:

```bash
bash docker/build.sh
```

To clone the repository with the necessary submodules (including a custom fork of `ros_gz`), use:

```bash
git clone --recurse-submodules https://github.com/mttga/pylrauv.git
```
