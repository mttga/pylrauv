import os
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Tuple, Union, Dict
import functools
import distrax
import math
from safetensors.flax import save_file, load_file
from flax.traverse_util import flatten_dict, unflatten_dict


def save_params(params: Dict, filename: Union[str, os.PathLike]) -> None:
    flattened_dict = flatten_dict(params, sep=",")
    save_file(flattened_dict, filename)


def load_params(filename: Union[str, os.PathLike]) -> Dict:
    flattened_dict = load_file(filename)
    return unflatten_dict(flattened_dict, sep=",")


# PPO Classes
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: Sequence[int]
    hidden_size: int

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones, avail_actions = x
        embedding = nn.Dense(
            self.hidden_size,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0)
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        unavail_actions = 1 - avail_actions
        action_logits = actor_mean - (unavail_actions * 1e10)

        pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


class CentralizedActorRNN:

    def __init__(
        self,
        seed,
        agent_params,
        agent_list,
        landmark_list,
        action_dim=5,
        hidden_dim=128,
        num_envs=1,
        pos_norm=1e-3,
    ):
        self.agent_params = agent_params
        self.agent_list = agent_list
        self.landmark_list = landmark_list
        self.num_agents = len(agent_list)
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.pos_norm = pos_norm
        self.num_envs = num_envs
        self.rng = jax.random.PRNGKey(seed)

        self.actor = ActorRNN(action_dim=self.action_dim, hidden_size=hidden_dim)

    def reset(self, seed=None):
        self.hidden = ScannedRNN.initialize_carry(self.num_agents, self.hidden_dim)
        if seed is not None:
            self.rng = jax.random.PRNGKey(seed)
        self.rng, key = jax.random.split(self.rng)

    def step(self, obs, done, avail_actions=None):

        if avail_actions is None:
            # TODO: available actions should be given by the environment
            avail_actions = jnp.ones((self.num_agents, self.action_dim), dtype=bool)

        # assumes a single boolean done signal
        dones = jnp.full((self.num_agents,), done, dtype=bool)

        self.rng, key = jax.random.split(self.rng)
        obs = {agent: self.preprocess_obs(o) for agent, o in obs.items()}
        obs = batchify(obs, self.agent_list, self.num_agents)

        ac_in = (
            obs[np.newaxis, ...],
            dones[np.newaxis, ...],
            avail_actions,
        )
        self.hidden, pi = self.actor.apply(self.agent_params, self.hidden, ac_in)

        action = pi.sample(seed=key)
        action = unbatchify(action, self.agent_list, self.num_envs, self.num_agents)
        action = {k: int(v.squeeze()) for k, v in action.items()}

        return action

    def preprocess_obs(self, obs):
        """
        transforms the observation from the gazebo environment to the format of the jax simplified env
        """

        new_obs = [
            obs["x"] * self.pos_norm,
            obs["y"] * self.pos_norm,
            obs["z"] * self.pos_norm,
            obs["rph_x"]
            * self.pos_norm,  # angle, could be also np.arctan(obs['vel_x'], obs['vel_y']),
            1,  # is agent
            1,  # is self
        ]  # self

        # other agents
        for agent in self.agent_list[:-1]:
            new_obs.extend(
                [
                    obs[f"{agent}_dx"] * self.pos_norm,
                    obs[f"{agent}_dy"] * self.pos_norm,
                    obs[f"{agent}_dz"] * self.pos_norm,
                    math.sqrt(
                        obs[f"{agent}_dx"] ** 2
                        + obs[f"{agent}_dy"] ** 2
                        + obs[f"{agent}_dz"]
                    )
                    * self.pos_norm,
                    1,  # is agent
                    0,  # is self
                ]
            )

        # landmarks
        for landmark in self.landmark_list:
            new_obs.extend(
                [
                    (obs[f"{landmark}_tracking_x"] - obs["x"])
                    * self.pos_norm,  # prediction relative to agent
                    (obs[f"{landmark}_tracking_y"] - obs["y"]) * self.pos_norm,
                    (obs[f"{landmark}_tracking_z"] - obs["z"]) * self.pos_norm,
                    (obs[f"{landmark}_range"]) * self.pos_norm,
                    0,  # is agent
                    0,  # is self
                ]
            )

        return jnp.array(new_obs)
