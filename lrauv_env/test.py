import time
import numpy as np
from env import LrauvEnv

def check_relative_distances(agents_data, error_margin=0.5):
    """check if the relative distances between two agents are similar"""
    agent_keys = list(agents_data.keys())

    for i, agent_i_key in enumerate(agent_keys):
        for agent_j_key in agent_keys[i + 1:]:
            dx_i_to_j = agents_data[agent_i_key][f"{agent_j_key}_dx"]
            dx_j_to_i = agents_data[agent_j_key][f"{agent_i_key}_dx"]
            diff = abs(dx_i_to_j - dx_j_to_i)

            if diff > error_margin:
                error_msg = f"Distance mismatch between {agent_i_key} and {agent_j_key}: {agent_i_to_j_dx} vs {agent_j_to_i_dx}. Difference: {diff}"
                raise ArithmeticError(error_msg)
            
def check_init_positions(state, min_distance=20, max_distance_from_agent=100):
    positions = np.array([[v['x'], v['y'], v['z']] for v in state.values()])
    agent_indices = [i for i, k in enumerate(state.keys()) if 'agent' in k]

    distances = np.sqrt(np.sum((positions[:, np.newaxis] - positions) ** 2, axis=2))
    np.fill_diagonal(distances, np.inf)  # set the diagonal to infinity to ignore self-distances

    # Check if every vehicle is more far away than the minimum distance threshold
    if np.any(distances < min_distance):
        i, j = np.where(distances < min_distance)
        raise ValueError(f"{list(state.keys())[i[0]]} is too close to {list(state.keys())[j[0]]}: {distances[i[0], j[0]]} < {min_distance}")

    # Check if every agent is not too far from any other vehicle
    np.fill_diagonal(distances, 0.)
    agent_distances = distances[agent_indices, :]
    if np.any(agent_distances > max_distance_from_agent):
        i, j = np.where(agent_distances > max_distance_from_agent)
        raise ValueError(f"Agent {list(state.keys())[agent_indices[i[0]]]} is too far from {list(state.keys())[j[0]]}: {agent_distances[i[0], j[0]]} > {max_distance_from_agent}")

    # Calculate the average distance between any vehicle and the agents
    avg_agent_distance = np.mean(agent_distances)

    return avg_agent_distance

def test_step():

    steps = 20
    step_time = 20
    env = LrauvEnv(n_agents=3, n_landmarks=3, render=True, prop_range_agent=(5., 5.), prop_range_landmark=(5., 5.))
    t0 = time.time()
    
    state = env.reset()
    print('Initial state:', state)
    
    all_obs = []
    for i in range(steps):
        print('step', i)
        obs, state = env.step(step_time=step_time)
        for k, v in obs.items():
            print('obs',k, v)
            print('\n')
        all_obs.append(obs)

    time_for_completion = time.time()-t0
    print(f'Time for completing {steps} steps: {time_for_completion:.2f}s', )
    print(f'Final simulation time:', env.sim_time)
    print(f'Real time speed up:', env.sim_time / time_for_completion)

    env.close()

    # check if the final simulation time is as expected
    assert steps*step_time - 2 <= env.sim_time <= steps*step_time + 2, \
        f"Final simulation time should be {steps*step_time} but is {env.sim_time}"
    
    for o in all_obs:
        check_relative_distances(o) 

    print("Test passed")

def test_init():
    # initialize the environment with reset several times and prints the average time for inizialization
    times = []
    n_exp = 5
    min_distance = 20
    max_distance = 100
    env = LrauvEnv(n_agents=3, n_landmarks=3, render=True, min_distance=min_distance, max_distance=max_distance)
    avg_distances = []
    for _ in  range(n_exp):
        t0 = time.time()
        obs, state = env.reset()
        init_time = time.time()-t0
        times.append(init_time)
        avg_distances.append(check_init_positions(state, min_distance, max_distance_from_agent=max_distance))
        print('Initial obs:', obs)
        print('Initial state:', state)
        print(f'Time for completing initialization: {init_time:.2f}')

    env.close()
    print(f'Average time for initialization: {np.mean(times):.2f}')
    print(f'Average distance between agents and any other entity: {np.mean(avg_distances):.2f}')
    print("Test passed")

if __name__=='__main__':
    test_step()