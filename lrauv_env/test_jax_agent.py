import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import argparse
from env import LrauvEnv
from wrappers.jax_agent import CentralizedActorRNN, load_params

def test_episode(steps=128, step_time=30, print_obs=False, print_state=False, render=True, model_path=None):
    env = LrauvEnv(
        n_agents=2,
        n_landmarks=2,
        render=render,
        prop_range_agent=(30.0, 30.0),
        prop_range_landmark=(0.0, 10.0),
        dirchange_time_range_landmark=(2, 10),
        tracking_method="pf",
        agent_controller="rudder_discrete",
        landmark_controller="linear_random",
    )

    params = load_params(model_path)

    if 'ppo' in model_path and 'transformer' in model_path:
        agent = CentralizedActorRNN(
            seed=0,
            agent_params=params['actor'],
            agent_list=env.agents_ids,
            landmark_list=env.landmarks_ids,
            action_dim=5,
            hidden_dim=64,
            pos_norm=1e-3,
            agent_class='ppo_transformer',
            mask_ranges=True,
            matrix_obs=True,
            add_agent_id=False,
            num_layers=2,
            num_heads=8,
            ff_dim=128,
        )
    elif 'ppo' in model_path:
        agent = CentralizedActorRNN(
            seed=0,
            agent_params=params['actor'],
            agent_list=env.agents_ids,
            landmark_list=env.landmarks_ids,
            action_dim=5,
            hidden_dim=128,
            pos_norm=1e-3,
            agent_class='ppo_rnn',
            mask_ranges=True,
            add_agent_id=False,
            num_layers=2,
        )
    else:
        agent = CentralizedActorRNN(
            seed=0,
            agent_params=params,
            agent_list=env.agents_ids,
            landmark_list=env.landmarks_ids,
            action_dim=5,
            hidden_dim=256,
            pos_norm=1e-3,
            num_layers=4,
            ppo=False,
            add_agent_id=True,
        )

    agent.reset()

    t0 = time.time()

    obs, state = env.reset()
    states = [state]
    obss = [obs]
    actionss = []

    for i in range(steps):
        print(f"Step {i}")
        avail_actions = env.get_available_actions()
        actions = agent.step(obs, done=False, avail_actions=avail_actions)
        obs, state = env.step(actions=actions, step_time=step_time)
        if print_state:
            print("state", state)
        if print_obs:
            for k, v in obs.items():
                print("obs", k, v)
                print("\n")
        actionss.append(actions)
        states.append(state)
        obss.append(obs)

    time_for_completion = time.time() - t0
    print(f"Time for completing {steps} steps: {time_for_completion:.2f}s")
    print(f"Final simulation time:", env.sim_time)
    print(f"Real time speed up:", env.sim_time / time_for_completion)

    env.close()
    return obss, states, actionss

def plot_episode(states, output_path, name="episode"):
    os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)
    entities = list(states[0].keys())
    data_entities = {e: pd.DataFrame([s[e] for s in states]) for e in entities}
    for e in entities:
        plt.plot(data_entities[e]['x'], data_entities[e]['y'], label=e)
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "plots", f"{name}.png"))
    plt.close()

def collect(model_path, output_path, episodes=15, steps=128, step_time=30):
    data = []
    os.makedirs(output_path, exist_ok=True)
    t0 = time.time()
    for e in range(episodes):
        obss, states, actionss = test_episode(steps=steps, step_time=step_time, render=False, model_path=model_path)
        data.append({"obss": obss, "states": states, "actionss": actionss})
        plot_episode(states, output_path, name=f"episode_{e}")

    with open(os.path.join(output_path, 'jax_episodes.json'), 'w') as f:
        json.dump(data, f)

    time_for_completion = time.time() - t0
    print(f"Time for completing {episodes} episodes: {time_for_completion:.2f}s")

def main():
    parser = argparse.ArgumentParser(description="Run LRAUV environment in collect or test mode.")
    parser.add_argument("--mode", type=str, choices=["collect", "test"], default="test", help="Mode to run the script: collect or test.")
    parser.add_argument("--model_path", type=str, default="models/default_model.safetensors", help="Path to the model parameters.")
    parser.add_argument("--output_path", type=str, default="outputs/default_output", help="Path to save outputs.")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to collect or test.")
    parser.add_argument("--steps", type=int, default=128, help="Number of steps per episode.")
    parser.add_argument("--step_time", type=int, default=30, help="Time per step in seconds.")

    args = parser.parse_args()

    model_path = args.model_path
    model_path = "models/utracking_follow_crash_penalty/utracking_2_vs_2/mappo_transformer_follow_crash_penalty_difficult_from_2_vs_2_utracking_2_vs_2_seed0_vmap0.safetensors"

    if args.mode == "collect":
        collect(model_path, args.output_path, args.episodes, args.steps, args.step_time)
    elif args.mode == "test":
        for i in range(args.episodes):
            test_episode(steps=args.steps, step_time=args.step_time, model_path=model_path)

if __name__ == "__main__":
    main()
