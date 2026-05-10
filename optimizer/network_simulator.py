"""Gym-compatible telecom network simulator for optimization experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

np.random.seed(42)

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    gym = None

    class _FallbackEnv:
        """Minimal Env fallback used when gymnasium is not installed."""

        pass

    class _FallbackBox:
        """Minimal Box fallback exposing shape and dtype."""

        def __init__(self, low, high, dtype=np.float32):
            """Store bounds for compatibility with gymnasium spaces.Box."""
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = self.low.shape
            self.dtype = dtype

    class _FallbackDiscrete:
        """Minimal Discrete fallback exposing sample()."""

        def __init__(self, n: int):
            """Store the number of discrete actions."""
            self.n = int(n)

        def sample(self) -> int:
            """Return a random action index."""
            return int(np.random.randint(0, self.n))

    class _FallbackSpaces:
        """Namespace compatible with gymnasium.spaces."""

        Box = _FallbackBox
        Discrete = _FallbackDiscrete

    class _FallbackGym:
        """Namespace compatible with gymnasium."""

        Env = _FallbackEnv

    gym = _FallbackGym()
    spaces = _FallbackSpaces()


ACTIONS = [
    "reroute_traffic",
    "increase_bandwidth",
    "reduce_load",
    "rebalance_towers",
    "do_nothing",
]


def calc_congestion_prob(latency, packet_loss, throughput_mbps):
    prob = (
        (latency / 5000.0) * 0.4
        + (packet_loss / 1.0) * 0.4
        + (1.0 - min(throughput_mbps / 1000.0, 1.0)) * 0.2
    )
    return float(np.clip(prob, 0.0, 1.0))


@dataclass
class NetworkState:
    """Container for the simulator's operational state."""

    latency: float
    throughput: float
    packet_loss: float
    tower_loads: np.ndarray
    congestion_probability: float


class TelecomNetworkEnv(gym.Env):
    """Simulate telecom network transitions for RL-based optimization."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, initial_state: dict[str, Any] | None = None, n_towers: int = 5, max_steps: int = 30, seed: int = 42):
        """Initialize the network simulator with seeded randomness."""
        self.n_towers = int(n_towers)
        self.max_steps = int(max_steps)
        self.seed_value = int(seed)
        self.rng = np.random.default_rng(self.seed_value)
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=np.array([5.0, 10.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([5000.0, 1000.0, 1.0, 100.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.initial_state = initial_state
        self.current_step = 0
        self.state = self._state_from_dict(initial_state) if initial_state else self._random_state()

    def _state_from_dict(self, state_dict: dict[str, Any]) -> NetworkState:
        """Build a simulator state from a user-provided dictionary."""
        latency = float(np.clip(state_dict.get("latency", 80.0), 5.0, 5000.0))
        throughput = float(np.clip(state_dict.get("throughput", 450.0), 10.0, 1000.0))
        packet_loss = float(state_dict.get("packet_loss", 0.05))
        if packet_loss > 1.0:
            packet_loss /= 100.0
        packet_loss = float(np.clip(packet_loss, 0.0, 1.0))
        tower_load = state_dict.get("tower_load", state_dict.get("tower_loads", 55.0))
        if isinstance(tower_load, (list, tuple, np.ndarray)):
            tower_loads = np.asarray(tower_load, dtype=float)
            if tower_loads.size != self.n_towers:
                tower_loads = np.resize(tower_loads, self.n_towers)
        else:
            tower_loads = np.full(self.n_towers, float(tower_load), dtype=float)
        tower_loads = np.clip(tower_loads, 0.0, 100.0)
        congestion_probability = state_dict.get("congestion_probability", state_dict.get("congestion_prob"))
        if congestion_probability is None:
            congestion_probability = self.calculate_congestion_probability(latency, throughput, packet_loss, tower_loads)
        return NetworkState(latency, throughput, packet_loss, tower_loads, float(np.clip(congestion_probability, 0.0, 1.0)))

    def _random_state(self) -> NetworkState:
        """Create a randomized but realistic telecom state."""
        latency = float(self.rng.uniform(20.0, 5000.0))
        throughput = float(self.rng.uniform(10.0, 1000.0))
        packet_loss = float(self.rng.uniform(0.0, 1.0))
        tower_loads = self.rng.uniform(25.0, 95.0, size=self.n_towers)
        congestion_probability = self.calculate_congestion_probability(latency, throughput, packet_loss, tower_loads)
        return NetworkState(latency, throughput, packet_loss, tower_loads, congestion_probability)

    def _observation(self) -> np.ndarray:
        """Return the current observation vector."""
        return np.array(
            [
                self.state.latency,
                self.state.throughput,
                self.state.packet_loss,
                float(np.mean(self.state.tower_loads)),
                self.state.congestion_probability,
            ],
            dtype=np.float32,
        )

    def state_dict(self) -> dict[str, Any]:
        """Return the current simulator state as a dictionary."""
        return {
            "latency": float(self.state.latency),
            "throughput": float(self.state.throughput),
            "packet_loss": float(self.state.packet_loss),
            "tower_load": float(np.mean(self.state.tower_loads)),
            "tower_loads": [float(value) for value in self.state.tower_loads],
            "congestion_probability": float(self.state.congestion_probability),
            "congestion_prob": float(self.state.congestion_probability),
        }

    def calculate_congestion_probability(self, latency: float, throughput: float, packet_loss: float, tower_loads: np.ndarray) -> float:
        """Calculate congestion probability from latency, throughput, loss, and tower load."""
        return calc_congestion_prob(float(latency), float(packet_loss), float(throughput))

    def _action_name(self, action: int | str) -> str:
        """Normalize an action index or action name."""
        if isinstance(action, str):
            if action not in ACTIONS:
                raise ValueError(f"Unknown optimization action: {action}")
            return action
        return ACTIONS[int(action)]

    def _clip_state(self) -> None:
        """Clamp simulator state to dashboard-compatible operational ranges."""
        self.state.latency = float(np.clip(self.state.latency, 5.0, 5000.0))
        self.state.throughput = float(np.clip(self.state.throughput, 10.0, 1000.0))
        self.state.packet_loss = float(np.clip(self.state.packet_loss, 0.0, 1.0))
        self.state.tower_loads = np.clip(self.state.tower_loads, 10.0, 100.0)

    def _apply_action(self, action: int | str) -> None:
        """Apply the selected optimization action to the current state."""
        action_name = self._action_name(action)
        if action_name == "reroute_traffic":
            self.state.latency *= float(self.rng.uniform(0.60, 0.75))
            self.state.packet_loss *= float(self.rng.uniform(0.80, 0.90))
            self.state.throughput *= float(self.rng.uniform(1.05, 1.15))
        elif action_name == "increase_bandwidth":
            self.state.throughput *= float(self.rng.uniform(1.30, 1.50))
            self.state.latency *= float(self.rng.uniform(0.90, 1.00))
        elif action_name == "reduce_load":
            self.state.packet_loss *= float(self.rng.uniform(0.50, 0.70))
            self.state.latency *= float(self.rng.uniform(0.85, 0.95))
            self.state.throughput *= float(self.rng.uniform(1.05, 1.15))
        elif action_name == "rebalance_towers":
            self.state.tower_loads = np.maximum(self.state.tower_loads * float(self.rng.uniform(0.60, 0.75)), 30.0)
            self.state.latency *= float(self.rng.uniform(0.85, 0.95))
        elif action_name == "do_nothing":
            self.state.latency *= float(self.rng.uniform(0.98, 1.02))
        self._clip_state()

    def _apply_natural_drift(self) -> None:
        """Apply natural network drift and measurement noise."""
        self.state.latency += float(self.rng.normal(1.5, 4.0))
        self.state.throughput += float(self.rng.normal(-5.0, 18.0))
        self.state.packet_loss += float(self.rng.normal(0.15, 0.6))
        self.state.tower_loads += self.rng.normal(0.3, 2.2, size=self.n_towers)
        self.state.latency = float(np.clip(self.state.latency, 20.0, 200.0))
        self.state.throughput = float(np.clip(self.state.throughput, 10.0, 1000.0))
        self.state.packet_loss = float(np.clip(self.state.packet_loss, 0.0, 20.0))
        self.state.tower_loads = np.clip(self.state.tower_loads, 0.0, 100.0)

    def _reward(self, old_state: dict[str, Any], action_name: str) -> float:
        """Compute reward from actual before/after improvement."""
        old_congestion = float(old_state.get("congestion_prob", old_state.get("congestion_probability", 0.5)))
        new_congestion = float(self.state.congestion_probability)
        reward = 0.0
        healthy_idle = action_name == "do_nothing" and new_congestion <= 0.08
        if healthy_idle:
            reward += 10.0
        elif new_congestion < old_congestion:
            reward += (old_congestion - new_congestion) * 50.0
        else:
            reward -= 2.0

        if self.state.latency < float(old_state.get("latency", 100.0)):
            reward += 2.0

        if self.state.throughput > float(old_state.get("throughput", 500.0)):
            reward += 1.0

        if action_name == "do_nothing" and new_congestion > 0.08:
            reward -= 3.0

        return float(reward)

    def step(self, action: int | str):
        """Advance the simulator by one action."""
        old_state = self.state_dict().copy()
        action_name = self._action_name(action)
        self._apply_action(action_name)
        self.state.congestion_probability = self.calculate_congestion_probability(
            self.state.latency,
            self.state.throughput,
            self.state.packet_loss,
            self.state.tower_loads,
        )
        self.current_step += 1
        reward = self._reward(old_state, action_name)
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = self.state_dict()
        info["action_name"] = action_name
        return self._observation(), reward, terminated, truncated, info

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the simulator and return the initial observation."""
        if seed is not None:
            self.seed_value = int(seed)
            self.rng = np.random.default_rng(self.seed_value)
        self.current_step = 0
        state_override = options.get("initial_state") if options else None
        if state_override is not None:
            self.state = self._state_from_dict(state_override)
        elif self.initial_state is not None:
            self.state = self._state_from_dict(self.initial_state)
        else:
            self.state = self._random_state()
        return self._observation(), self.state_dict()

    def render(self) -> str:
        """Render the current simulator state as a readable string."""
        state = self.state_dict()
        return (
            f"latency={state['latency']:.1f}ms, throughput={state['throughput']:.1f}Mbps, "
            f"packet_loss={state['packet_loss']:.2f}%, tower_load={state['tower_load']:.1f}%, "
            f"congestion_prob={state['congestion_probability']:.2f}"
        )
