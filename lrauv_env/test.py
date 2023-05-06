import time
import numpy as np
from env import LrauvEnv

def test_step():

    steps = 5
    step_time = 30
    env = LrauvEnv(n_agents=3, n_landmarks=3, use_gui=True)
    t0 = time.time()
    
    state = env.reset()
    print('Initial state:', state)
    
    for i in range(steps):
        print('step', i)
        state = env.step(step_time=step_time)
        print(state)

    time_for_completion = time.time()-t0
    print(f'Time for completing {steps} steps: {time_for_completion:.2f}s', )
    print(f'Final simulation time:', env.sim_time)
    print(f'Real time speed up:', env.sim_time / time_for_completion)

    # check if the final simulation time is as expected
    assert steps*step_time - 1 <= env.sim_time <= steps*step_time + 1, \
        f"Final simulation time should be {steps*step_time} but is {env.sim_time}"
    
    env.close()

def test_init():
    # initialize the environment with reset several times and prints the average time for inizialization
    times = []
    n_exp = 5
    env = LrauvEnv(n_agents=3, n_landmarks=3, use_gui=True)
    for _ in  range(n_exp):
        t0 = time.time()
        state = env.reset()
        init_time = time.time()-t0
        times.append(init_time)
        print('Initial state:', state)
        print(f'Time for completing initialization: {init_time:.2f}')
    env.close()
    print(f'Average time for initialization: {np.mean(times):.2f}')

if __name__=='__main__':
    test_step()