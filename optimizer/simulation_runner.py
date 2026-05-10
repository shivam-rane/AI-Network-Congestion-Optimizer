"""Sequential optimization simulation runner."""

from __future__ import annotations

from typing import Any

import numpy as np

from optimizer.network_simulator import ACTIONS, TelecomNetworkEnv, calc_congestion_prob
from optimizer.rl_agent import get_optimal_action

np.random.seed(42)


def _extract_state_from_info(info: dict[str, Any]) -> dict[str, Any]:
    """Extract agent-compatible state fields from simulator info."""
    return {
        "latency": float(info["latency"]),
        "throughput": float(info["throughput"]),
        "packet_loss": float(info["packet_loss"]),
        "tower_load": float(info["tower_load"]),
        "tower_loads": info.get("tower_loads", []),
        "congestion_probability": float(info["congestion_probability"]),
        "congestion_prob": float(info.get("congestion_prob", info["congestion_probability"])),
    }


def _normalized_initial_state(initial_state: dict[str, Any]) -> dict[str, float]:
    """Normalize dashboard inputs into simulator units after env reset."""
    latency = float(initial_state.get("latency", 100.0))
    throughput = float(initial_state.get("throughput", 200.0))
    packet_loss = float(initial_state.get("packet_loss", 0.05))
    tower_load = float(initial_state.get("tower_load", 60.0))

    if 0.0 <= throughput <= 1.0:
        throughput *= 1000.0
    if packet_loss > 1.0:
        packet_loss /= 100.0

    fallback_congestion = calc_congestion_prob(latency, packet_loss, throughput)
    congestion_prob = float(
        initial_state.get(
            "congestion_prob",
            initial_state.get("congestion_probability", fallback_congestion),
        )
    )

    return {
        "latency": float(np.clip(latency, 5.0, 5000.0)),
        "throughput": float(np.clip(throughput, 10.0, 1000.0)),
        "packet_loss": float(np.clip(packet_loss, 0.0, 1.0)),
        "tower_load": float(np.clip(tower_load, 10.0, 100.0)),
        "congestion_probability": float(np.clip(congestion_prob, 0.0, 1.0)),
        "congestion_prob": float(np.clip(congestion_prob, 0.0, 1.0)),
    }


def summarize_simulation(simulation_log: list[dict[str, Any]], initial_state: dict[str, Any]) -> dict[str, Any]:
    """Compute summary metrics for an optimization log."""
    if not simulation_log:
        return {
            "total_reward": 0.0,
            "congestion_reduction_pct": 0.0,
            "latency_improvement_pct": 0.0,
            "throughput_improvement_pct": 0.0,
            "recommended_action_sequence": [],
        }

    final_state = simulation_log[-1]
    initial_latency = float(initial_state.get("latency", final_state["latency"]))
    initial_throughput = float(initial_state.get("throughput", final_state["throughput"]))
    initial_packet_loss = float(initial_state.get("packet_loss", final_state["packet_loss"]))
    initial_congestion = calc_congestion_prob(initial_latency, initial_packet_loss, initial_throughput)
    final_congestion = float(final_state["congestion_prob"])
    final_latency = float(final_state["latency"])
    final_throughput = float(final_state["throughput"])

    return {
        "total_reward": float(sum(row["reward"] for row in simulation_log)),
        "congestion_reduction_pct": float(((initial_congestion - final_congestion) / max(initial_congestion, 1e-6)) * 100.0),
        "latency_improvement_pct": float(((initial_latency - final_latency) / max(initial_latency, 1e-6)) * 100.0),
        "throughput_improvement_pct": float(((final_throughput - initial_throughput) / max(initial_throughput, 1e-6)) * 100.0),
        "recommended_action_sequence": [row["action"] for row in simulation_log if row["step"] > 0],
    }


def run_optimization_simulation(initial_state: dict[str, Any], n_steps: int = 10) -> dict[str, Any]:
    """Run RL-guided optimization and return the log plus summary metrics."""
    normalized_initial = _normalized_initial_state(initial_state)
    env = TelecomNetworkEnv(max_steps=n_steps, seed=42)
    env.reset()
    env.state = env._state_from_dict(normalized_initial)
    current_state = _extract_state_from_info(env.state_dict())
    simulation_log = [
        {
            "step": 0,
            "action": "initial",
            "latency": round(current_state["latency"], 2),
            "throughput": round(current_state["throughput"], 2),
            "packet_loss": round(current_state["packet_loss"], 4),
            "tower_load": round(current_state["tower_load"], 2),
            "congestion_prob": round(current_state["congestion_probability"], 4),
            "reward": 0.0,
            "expected_reward": 0.0,
        }
    ]

    for step in range(1, int(n_steps) + 1):
        before = current_state.copy()
        action_name = get_optimal_action(before)
        _, reward, terminated, truncated, info = env.step(action_name)
        current_state = _extract_state_from_info(info)
        simulation_log.append(
            {
                "step": step,
                "action": action_name,
                "latency": round(current_state["latency"], 2),
                "throughput": round(current_state["throughput"], 2),
                "packet_loss": round(current_state["packet_loss"], 4),
                "tower_load": round(current_state["tower_load"], 2),
                "congestion_prob": round(current_state["congestion_probability"], 4),
                "reward": round(float(reward), 3),
                "expected_reward": round(float(reward), 3),
                "latency_before": round(before["latency"], 2),
                "throughput_before": round(before["throughput"], 2),
                "packet_loss_before": round(before["packet_loss"], 4),
                "tower_load_before": round(before["tower_load"], 2),
                "congestion_prob_before": round(before["congestion_probability"], 4),
            }
        )
        if terminated or truncated:
            break

    return {
        "log": simulation_log,
        "simulation_log": simulation_log,
        "summary": summarize_simulation(simulation_log, normalized_initial),
    }
