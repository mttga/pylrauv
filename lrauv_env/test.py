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

def test_step():

    steps = 5
    step_time = 30
    env = LrauvEnv(n_agents=3, n_landmarks=3, use_gui=True)
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
    env = LrauvEnv(n_agents=3, n_landmarks=3, use_gui=True)
    for _ in  range(n_exp):
        t0 = time.time()
        obs, state = env.reset()
        init_time = time.time()-t0
        times.append(init_time)
        print('Initial obs:', obs)
        print('Initial state:', state)
        print(f'Time for completing initialization: {init_time:.2f}')
    env.close()
    print(f'Average time for initialization: {np.mean(times):.2f}')

if __name__=='__main__':
    test_step()