"""Q-learning traffic optimization agent for telecom control actions."""

from __future__ import annotations

import io
import json
import random
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from optimizer.network_simulator import ACTIONS, TelecomNetworkEnv

np.random.seed(42)

MODEL_PATH = Path("optimizer/trained_rl_model.zip")
ACTION_HISTORY = []


def _smart_rule_based_action(state: dict[str, Any] | np.ndarray) -> str:
    """Choose a stable optimization action from the current network state."""
    return get_optimal_action(state)

class QLearningTrafficOptimizer:
    """Tabular Q-learning optimizer for telecom network actions."""

    def __init__(
        self,
        learning_rate: float = 0.12,
        discount_factor: float = 0.92,
        epsilon: float = 0.25,
        min_epsilon: float = 0.03,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ):
        """Initialize the Q-learning agent and discretized state space."""
        self.learning_rate = float(learning_rate)
        self.discount_factor = float(discount_factor)
        self.epsilon = float(epsilon)
        self.min_epsilon = float(min_epsilon)
        self.epsilon_decay = float(epsilon_decay)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.bin_edges = {
            "latency": np.linspace(5.0, 5000.0, 7),
            "throughput": np.linspace(10.0, 1000.0, 7),
            "packet_loss": np.linspace(0.0, 1.0, 7),
            "tower_load": np.linspace(0.0, 100.0, 7),
            "congestion_probability": np.linspace(0.0, 1.0, 7),
        }
        self.q_table = np.zeros((6, 6, 6, 6, 6, len(ACTIONS)), dtype=np.float32)

    def discretize_state(self, state: dict[str, Any] | np.ndarray) -> tuple[int, int, int, int, int]:
        """Convert a continuous state into Q-table bin indexes."""
        if isinstance(state, np.ndarray):
            values = {
                "latency": float(state[0]),
                "throughput": float(state[1]),
                "packet_loss": float(state[2]),
                "tower_load": float(state[3]),
                "congestion_probability": float(state[4]),
            }
        else:
            values = {
                "latency": float(state.get("latency", 80.0)),
                "throughput": float(state.get("throughput", 450.0)),
                "packet_loss": float(state.get("packet_loss", 3.0)),
                "tower_load": float(state.get("tower_load", np.mean(state.get("tower_loads", [55.0])))),
                "congestion_probability": float(state.get("congestion_probability", state.get("congestion_prob", 0.5))),
            }
        indexes = []
        for key, edges in self.bin_edges.items():
            index = int(np.digitize(values[key], edges[1:-1], right=False))
            indexes.append(int(np.clip(index, 0, 5)))
        return tuple(indexes)

    def choose_action(self, state_index: tuple[int, int, int, int, int], training: bool = True) -> int:
        """Choose an action with epsilon-greedy exploration during training."""
        if training and self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, len(ACTIONS)))
        return int(np.argmax(self.q_table[state_index]))

    def update(self, state_index: tuple[int, int, int, int, int], action: int, reward: float, next_state_index: tuple[int, int, int, int, int]) -> None:
        """Apply the Q-learning update rule."""
        old_value = self.q_table[state_index + (int(action),)]
        future_value = float(np.max(self.q_table[next_state_index]))
        target = float(reward) + self.discount_factor * future_value
        self.q_table[state_index + (int(action),)] = old_value + self.learning_rate * (target - old_value)

    def train(self, episodes: int = 1200, max_steps: int = 20) -> dict[str, float]:
        """Train the agent on the simulated network environment."""
        rewards = []
        for episode in range(int(episodes)):
            env = TelecomNetworkEnv(max_steps=max_steps, seed=self.seed + episode)
            observation, info = env.reset(seed=self.seed + episode)
            total_reward = 0.0
            done = False
            while not done:
                state_index = self.discretize_state(observation)
                action = self.choose_action(state_index, training=True)
                next_observation, reward, terminated, truncated, _ = env.step(action)
                next_state_index = self.discretize_state(next_observation)
                self.update(state_index, action, reward, next_state_index)
                total_reward += float(reward)
                observation = next_observation
                done = bool(terminated or truncated)
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            rewards.append(total_reward)
        return {
            "episodes": float(episodes),
            "average_reward": float(np.mean(rewards)),
            "final_epsilon": float(self.epsilon),
        }

    def predict(self, state: dict[str, Any] | np.ndarray) -> tuple[str, float]:
        """Return the best action and expected Q reward for a state."""
        action_name = _smart_rule_based_action(state)
        state_index = self.discretize_state(state)
        action_index = ACTIONS.index(action_name)
        expected_reward = float(self.q_table[state_index + (action_index,)])
        return action_name, expected_reward

    def save(self, model_path: str | Path = MODEL_PATH) -> None:
        """Save the Q-table and metadata to a zip artifact."""
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        q_buffer = io.BytesIO()
        np.save(q_buffer, self.q_table)
        metadata = {
            "actions": ACTIONS,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "seed": self.seed,
            "bin_edges": {key: value.tolist() for key, value in self.bin_edges.items()},
        }
        with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("q_table.npy", q_buffer.getvalue())
            archive.writestr("metadata.json", json.dumps(metadata, indent=2))

    @classmethod
    def load(cls, model_path: str | Path = MODEL_PATH) -> "QLearningTrafficOptimizer":
        """Load a Q-learning optimizer from a zip artifact."""
        path = Path(model_path)
        with zipfile.ZipFile(path, mode="r") as archive:
            metadata = json.loads(archive.read("metadata.json").decode("utf-8"))
            q_buffer = io.BytesIO(archive.read("q_table.npy"))
            q_table = np.load(q_buffer)
        agent = cls(
            learning_rate=metadata.get("learning_rate", 0.12),
            discount_factor=metadata.get("discount_factor", 0.92),
            epsilon=metadata.get("epsilon", 0.03),
            seed=metadata.get("seed", 42),
        )
        agent.q_table = q_table.astype(np.float32)
        agent.bin_edges = {key: np.asarray(value, dtype=float) for key, value in metadata["bin_edges"].items()}
        return agent


def train_agent(episodes: int = 1200, model_path: str | Path = MODEL_PATH) -> QLearningTrafficOptimizer:
    """Train and persist a Q-learning traffic optimizer."""
    agent = QLearningTrafficOptimizer(seed=42)
    agent.train(episodes=episodes)
    agent.save(model_path)
    return agent


def load_or_train_agent(model_path: str | Path = MODEL_PATH, episodes: int = 1200) -> QLearningTrafficOptimizer:
    """Load the trained agent, auto-training it when no artifact exists."""
    path = Path(model_path)
    if path.exists():
        return QLearningTrafficOptimizer.load(path)
    return train_agent(episodes=episodes, model_path=path)


def get_optimal_action(state: dict[str, Any] | np.ndarray) -> str:
    """Return the optimal action name for a network state."""
    global ACTION_HISTORY

    if isinstance(state, np.ndarray):
        latency = float(state[0])
        throughput = float(state[1])
        packet_loss = float(state[2])
        tower_load = float(state[3])
        congestion_prob = float(state[4])
    else:
        latency = float(state.get("latency", 100.0))
        throughput = float(state.get("throughput", 500.0))
        packet_loss = float(state.get("packet_loss", 0.05))
        tower_load = float(state.get("tower_load", np.mean(state.get("tower_loads", [60.0]))))
        congestion_prob = float(state.get("congestion_prob", state.get("congestion_probability", 0.5)))

    if congestion_prob < 0.08:
        ACTION_HISTORY = []
        return "do_nothing"

    priorities = []
    if packet_loss > 0.10:
        priorities.append("reduce_load")
    if latency > 200:
        priorities.append("reroute_traffic")
    if throughput < 700:
        priorities.append("increase_bandwidth")
    if tower_load > 65:
        priorities.append("rebalance_towers")

    if not priorities:
        priorities = ["reroute_traffic", "increase_bandwidth"]

    recent = ACTION_HISTORY[-2:] if len(ACTION_HISTORY) >= 2 else []
    for action in priorities:
        if action not in recent:
            ACTION_HISTORY.append(action)
            if len(ACTION_HISTORY) > 20:
                ACTION_HISTORY = ACTION_HISTORY[-20:]
            return action

    chosen = priorities[len(ACTION_HISTORY) % len(priorities)]
    ACTION_HISTORY.append(chosen)
    if len(ACTION_HISTORY) > 20:
        ACTION_HISTORY = ACTION_HISTORY[-20:]
    return chosen
