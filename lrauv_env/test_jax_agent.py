import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import json
import os
import argparse
from env import LrauvEnv
from wrappers.jax_agent import CentralizedActorRNN, load_params
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jax_manager")


def test_episode(
    num_agents=2,
    num_landmarks=2,
    steps=128,
    step_time=30,
    print_obs=False,
    print_state=False,
    render=True,
    model_path=None,
):
    env = LrauvEnv(
        n_agents=num_agents,
        n_landmarks=num_landmarks,
        render=render,
        prop_range_agent=(30.0, 30.0),
        prop_range_landmark=(0.0, 10.0),
        dirchange_time_range_landmark=(2, 10),
        max_distance=200,
        tracking_method="pf",
        agent_controller="rudder_discrete",
        landmark_controller="linear_random",
    )

    params = load_params(model_path)

    if "ppo" in model_path and "transformer" in model_path:
        agent = CentralizedActorRNN(
            seed=0,
            agent_params=params["actor"],
            agent_list=env.agents_ids,
            landmark_list=env.landmarks_ids,
            action_dim=5,
            hidden_dim=64,
            pos_norm=1e-3,
            agent_class="ppo_transformer",
            mask_ranges=True,
            matrix_obs=True,
            add_agent_id=False,
            num_layers=2,
            num_heads=8,
            ff_dim=128,
        )
    elif "ppo" in model_path:
        agent = CentralizedActorRNN(
            seed=0,
            agent_params=params["actor"],
            agent_list=env.agents_ids,
            landmark_list=env.landmarks_ids,
            action_dim=5,
            hidden_dim=128,
            pos_norm=1e-3,
            agent_class="ppo_rnn",
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
        logger.info(f"Step {i}")
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
    logger.info(f"Time for completing {steps} steps: {time_for_completion:.2f}s")
    logger.info(f"Final simulation time:", env.sim_time)
    logger.info(f"Real time speed up:", env.sim_time / time_for_completion)

    env.close()
    return obss, states, actionss


def plot_episode(states, output_path, name="episode"):
    os.makedirs(os.path.join(output_path, "plots"), exist_ok=True)
    entities = list(states[0].keys())
    data_entities = {e: pd.DataFrame([s[e] for s in states]) for e in entities}
    for e in entities:
        plt.plot(data_entities[e]["x"], data_entities[e]["y"], label=e)
        if 'landmark' in e:
            plt.plot(data_entities[e]["tracking_x"], data_entities[e]["tracking_y"], label=f'tracking_{e}')
    plt.legend()
    os.makedirs(output_path, exist_ok=True)
    plt.savefig(os.path.join(output_path, "plots", f"{name}.png"))
    plt.close()


def collect(model_path, output_path, episodes=15, id=0, **episode_kwargs):
    data = []
    os.makedirs(output_path, exist_ok=True)
    t0 = time.time()
    for e in range(episodes):
        obss, states, actionss = test_episode(
            render=False, model_path=model_path, **episode_kwargs
        )
        data.append({"obss": obss, "states": states, "actionss": actionss})

        # Save data
        with open(os.path.join(output_path, f"jax_episodes_{id}.json"), "w") as f:
            json.dump(data, f)

        plot_episode(states, output_path, name=f"episode_{e}_{id}")

    time_for_completion = time.time() - t0
    logger.info(f"Time for completing {episodes} episodes: {time_for_completion:.2f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Run LRAUV environment in collect or test mode."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["collect", "test"],
        default="test",
        help="Mode to run the script: collect or test.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/default_model.safetensors",
        help="Path to the model parameters.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/default_output",
        help="Path to save outputs.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of episodes to collect or test.",
    )
    parser.add_argument(
        "--steps", type=int, default=128, help="Number of steps per episode."
    )
    parser.add_argument(
        "--step_time", type=int, default=30, help="Time per step in seconds."
    )
    parser.add_argument("--num_agents", type=int, default=2, help="Number of Agents.")
    parser.add_argument(
        "--num_landmarks", type=int, default=2, help="Number of Landmarks"
    )
    parser.add_argument("--id", type=int, default=1, help="Id of the process.")

    args = parser.parse_args()


    if args.mode == "collect":
        collect(
            args.model_path,
            args.output_path,
            args.episodes,
            id=args.id,
            steps=args.steps,
            step_time=args.step_time,
            num_agents=args.num_agents,
            num_landmarks=args.num_landmarks,
        )
        
    elif args.mode == "test":
        for i in range(args.episodes):
            test_episode(
                steps=args.steps, step_time=args.step_time, model_path=args.model_path, num_agents=args.num_agents, num_landmarks=args.num_landmarks,
            )


if __name__ == "__main__":
    main()
