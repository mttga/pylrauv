"""
Helper script to get simple trajectory data from one agent and landmark using random actions.
"""
import time
import numpy as np
import json
import os
import sys
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from env import LrauvEnv


def main():

    steps = 200
    print_obs_state = True

    N = 5 # number of samples per combination
    thrusters = [5., 10., 20., 25.]
    step_times = [20, 30, 60]

    t0 = time.time()

    data = []
    for _ in range(N):
        for step_time in step_times:
            for t in thrusters:
                
                env = LrauvEnv(
                    n_agents=1,
                    n_landmarks=1,
                    render=False,
                    #rudder_range_landmark=(r, r),
                    prop_range_agent=(t, t),
                    prop_range_landmark=(t, t),
                    tracking_method='ls'
                )
        
                obs, state = env.reset()
                if print_obs_state:
                    print('Initial state:', state)
        
                all_obs = []
                all_states = []
                for i in range(steps):
                    print('step', i)
                    obs, state = env.step(step_time=step_time)
                    all_obs.append(obs)
                    all_states.append(state)

                env.close()

                pos_agent = np.array([[state['agent_1']['x'], state['agent_1']['y'], state['agent_1']['z']] for state in all_states])
                pos_land  = np.array([[state['landmark_1']['x'], state['landmark_1']['y'], state['landmark_1']['z']] for state in all_states])
                rph_agent = np.array([[state['agent_1']['rph_x'], state['agent_1']['rph_y']] for state in all_states])
                rph_land  = np.array([[state['landmark_1']['rph_x'], state['landmark_1']['rph_y']] for state in all_states])
                pqr_agent = np.array([[state['agent_1']['pqr_x'], state['agent_1']['pqr_y']] for state in all_states])
                pqr_land  = np.array([[state['landmark_1']['pqr_x'], state['landmark_1']['pqr_y']] for state in all_states])
                vel_agent = np.array([[state['agent_1']['vel_x'], state['agent_1']['vel_y'], state['agent_1']['vel_z']] for state in all_states])
                vel_land  = np.array([[state['landmark_1']['vel_x'], state['landmark_1']['vel_y'], state['landmark_1']['vel_z']] for state in all_states])
                rud_agent = np.array([state['agent_1']['rud_ang'] for state in all_states])
                rud_land  = np.array([state['landmark_1']['rud_ang'] for state in all_states])
                prop_agent = np.array([state['agent_1']['prop_vel'] for state in all_states])
                prop_land  = np.array([state['landmark_1']['prop_vel'] for state in all_states])

                data.append({
                    'dt':step_time,
                    'pos_agent':pos_agent.tolist(),
                    'pos_landmark':pos_land.tolist(),
                    'rph_agent':rph_agent.tolist(),
                    'rph_land':rph_land.tolist(),
                    'pqr_agent':pqr_agent.tolist(),
                    'pqr_land':pqr_land.tolist(),
                    'vel_agent':vel_agent.tolist(),
                    'vel_landmark':vel_land.tolist(),
                    'rud_agent':rud_agent.tolist(),
                    'rud_land':rud_land.tolist(),
                    'prop_agent':prop_agent.tolist(),
                    'prop_land':prop_land.tolist(),
                })
            
    time_for_completion = time.time()-t0
    print(f'Time for completing: {time_for_completion:.2f}s', )

    with open('trajectory_data.json', 'w') as f:
        json.dump(data, f)


if __name__=='__main__':
    main()