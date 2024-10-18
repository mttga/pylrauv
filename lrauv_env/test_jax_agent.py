import time
import numpy as np
from env import LrauvEnv
from wrappers.jax_agent import CentralizedActorRNN, load_params


def test_step():

    steps = 128
    step_time = 30
    print_obs_state = False
    env = LrauvEnv(
        n_agents=3,
        n_landmarks=3,
        render=True,
        prop_range_agent=(30.0, 30.0),
        prop_range_landmark=(0.0, 0.0),
        tracking_method="ls",
        agent_controller="rudder_discrete",
    )

    # load the agent params
    params = load_params(
        "models/mappo_rnn_3v3_shouldlearn_utracking_3_vs_3_seed0_vmap0.safetensors"
    )
    agent = CentralizedActorRNN(
        seed=0,
        agent_params=params["actor"],
        agent_list=env.agents_ids,
        landmark_list=env.landmarks_ids,
        action_dim=5,
        hidden_dim=128,
        pos_norm=1e-3,
    )
    agent.reset()

    t0 = time.time()

    obs, state = env.reset()

    for i in range(steps):
        print("step", i)

        # get the actions from the agent
        actions = agent.step(obs, done=False)
        obs, state = env.step(actions=actions, step_time=step_time)
        if print_obs_state:
            for k, v in obs.items():
                print("obs", k, v)
                print("\n")

    time_for_completion = time.time() - t0
    print(
        f"Time for completing {steps} steps: {time_for_completion:.2f}s",
    )
    print(f"Final simulation time:", env.sim_time)
    print(f"Real time speed up:", env.sim_time / time_for_completion)

    env.close()


if __name__ == "__main__":
    test_step()
