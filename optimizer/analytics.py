"""Optimization analytics and strategy comparison utilities."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from optimizer.network_simulator import ACTIONS, TelecomNetworkEnv
from optimizer.rl_agent import load_or_train_agent
from optimizer.simulation_runner import _extract_state_from_info

np.random.seed(42)


def _rule_based_action(state: dict[str, Any]) -> str:
    """Choose a baseline rule-based optimization action."""
    if state["tower_load"] >= 80.0 or state["congestion_probability"] >= 0.75:
        return "rebalance_towers"
    if state["packet_loss"] >= 8.0:
        return "reduce_load"
    if state["throughput"] <= 300.0:
        return "increase_bandwidth"
    if state["latency"] >= 100.0:
        return "reroute_traffic"
    return "do_nothing"


def _greedy_action(state: dict[str, Any], step: int) -> str:
    """Choose the action with the highest immediate simulated reward."""
    rewards = []
    for action_index, action_name in enumerate(ACTIONS):
        env = TelecomNetworkEnv(initial_state=state, max_steps=1, seed=4200 + step * 10 + action_index)
        env.reset(options={"initial_state": state})
        _, reward, _, _, _ = env.step(action_index)
        rewards.append((float(reward), action_name))
    rewards.sort(reverse=True, key=lambda item: item[0])
    return rewards[0][1]


def _simulate_strategy(initial_state: dict[str, Any], strategy: str, n_steps: int = 10) -> dict[str, Any]:
    """Simulate a named strategy and return final metrics."""
    agent = load_or_train_agent()
    rng = np.random.default_rng(42)
    env = TelecomNetworkEnv(initial_state=initial_state, max_steps=n_steps, seed=42)
    _, info = env.reset(options={"initial_state": initial_state})
    state = _extract_state_from_info(info)
    actions = []
    total_reward = 0.0

    for step in range(1, int(n_steps) + 1):
        if strategy == "RL Agent":
            action_name, _ = agent.predict(state)
        elif strategy == "Greedy":
            action_name = _greedy_action(state, step)
        elif strategy == "Random":
            action_name = ACTIONS[int(rng.integers(0, len(ACTIONS)))]
        elif strategy == "Rule-based":
            action_name = _rule_based_action(state)
        else:
            action_name = "do_nothing"

        _, reward, terminated, truncated, info = env.step(ACTIONS.index(action_name))
        state = _extract_state_from_info(info)
        actions.append(action_name)
        total_reward += float(reward)
        if terminated or truncated:
            break

    return {
        "final_congestion_prob": float(state["congestion_probability"]),
        "latency": float(state["latency"]),
        "throughput": float(state["throughput"]),
        "packet_loss": float(state["packet_loss"]),
        "tower_load": float(state["tower_load"]),
        "total_reward": float(total_reward),
        "actions": actions,
    }


def compare_strategies(initial_state: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Compare RL, greedy, random, and rule-based optimization strategies."""
    return {
        strategy: _simulate_strategy(initial_state, strategy, n_steps=10)
        for strategy in ["RL Agent", "Greedy", "Random", "Rule-based"]
    }


def plot_optimization_trajectory(simulation_log: list[dict[str, Any]]):
    """Plot congestion probability and latency across optimization steps."""
    steps = [row["step"] for row in simulation_log]
    congestion = [row["congestion_prob"] for row in simulation_log]
    latency = [row["latency"] for row in simulation_log]
    fig, ax1 = plt.subplots(figsize=(8, 3.8), facecolor="#0f172a")
    ax1.set_facecolor("#0f172a")
    ax1.plot(steps, congestion, marker="o", color="#22c55e", label="Congestion probability")
    ax1.set_xlabel("Step", color="#e5e7eb")
    ax1.set_ylabel("Congestion probability", color="#22c55e")
    ax1.tick_params(colors="#e5e7eb")
    ax1.grid(color="#334155", alpha=0.35)
    ax2 = ax1.twinx()
    ax2.plot(steps, latency, marker="s", color="#38bdf8", label="Latency")
    ax2.set_ylabel("Latency (ms)", color="#38bdf8")
    ax2.tick_params(colors="#e5e7eb")
    fig.tight_layout()
    return fig


def plot_strategy_comparison(comparison_dict: dict[str, dict[str, Any]]):
    """Plot strategy comparison across congestion, latency, and throughput KPIs."""
    strategies = list(comparison_dict.keys())
    metrics = [
        ("final_congestion_prob", "Congestion Prob", "#f97316"),
        ("latency", "Latency", "#38bdf8"),
        ("throughput", "Throughput", "#22c55e"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor="#0f172a")
    for ax, (metric, title, color) in zip(axes, metrics):
        values = [comparison_dict[strategy][metric] for strategy in strategies]
        ax.set_facecolor("#0f172a")
        ax.bar(strategies, values, color=color)
        ax.set_title(title, color="#f8fafc")
        ax.tick_params(axis="x", rotation=30, colors="#e5e7eb")
        ax.tick_params(axis="y", colors="#e5e7eb")
        ax.grid(axis="y", color="#334155", alpha=0.35)
        for spine in ax.spines.values():
            spine.set_color("#475569")
    fig.tight_layout()
    return fig
