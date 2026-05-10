"""Markdown report generation for network optimization runs."""

from __future__ import annotations

from typing import Any


def _format_value(value: Any) -> str:
    """Format a report value for markdown tables."""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _top_shap_features(shap_values: Any, limit: int = 5) -> list[tuple[str, float]]:
    """Normalize SHAP or feature-importance input into top feature rows."""
    if shap_values is None:
        return []
    if hasattr(shap_values, "to_dict"):
        rows = shap_values.to_dict(orient="records")
        feature_key = "feature"
        value_key = "importance" if rows and "importance" in rows[0] else "value"
        return [(str(row.get(feature_key, "unknown")), float(row.get(value_key, 0.0))) for row in rows[:limit]]
    if isinstance(shap_values, dict):
        items = sorted(shap_values.items(), key=lambda item: abs(float(item[1])), reverse=True)
        return [(str(key), float(value)) for key, value in items[:limit]]
    if isinstance(shap_values, list):
        normalized = []
        for row in shap_values[:limit]:
            if isinstance(row, dict):
                normalized.append((str(row.get("feature", "unknown")), float(row.get("importance", row.get("value", 0.0)))))
        return normalized
    return []


def _strategy_table(comparison_dict: dict[str, dict[str, Any]]) -> str:
    """Build a markdown table for strategy comparison results."""
    lines = [
        "| Strategy | Final Congestion | Latency | Throughput | Total Reward |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy, metrics in comparison_dict.items():
        lines.append(
            "| "
            + " | ".join(
                [
                    strategy,
                    _format_value(metrics.get("final_congestion_prob", 0.0)),
                    _format_value(metrics.get("latency", 0.0)),
                    _format_value(metrics.get("throughput", 0.0)),
                    _format_value(metrics.get("total_reward", 0.0)),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def generate_optimization_report(simulation_log: list[dict[str, Any]], comparison_dict: dict[str, dict[str, Any]], shap_values: Any) -> str:
    """Generate a markdown optimization report."""
    initial = simulation_log[0] if simulation_log else {}
    final = simulation_log[-1] if simulation_log else {}
    total_reward = sum(float(row.get("reward", 0.0)) for row in simulation_log)
    action_sequence = [row.get("action", "do_nothing") for row in simulation_log]
    top_features = _top_shap_features(shap_values)

    state_table = "\n".join(
        [
            "| Metric | Initial | Final |",
            "| --- | ---: | ---: |",
            f"| Latency | {_format_value(initial.get('latency', 0.0))} | {_format_value(final.get('latency', 0.0))} |",
            f"| Throughput | {_format_value(initial.get('throughput', 0.0))} | {_format_value(final.get('throughput', 0.0))} |",
            f"| Packet Loss | {_format_value(initial.get('packet_loss', 0.0))} | {_format_value(final.get('packet_loss', 0.0))} |",
            f"| Congestion Probability | {_format_value(initial.get('congestion_prob', 0.0))} | {_format_value(final.get('congestion_prob', 0.0))} |",
        ]
    )
    feature_lines = "\n".join([f"* {feature}: {value:.4f}" for feature, value in top_features]) or "* SHAP features unavailable"
    recommendation = "Continue monitoring."
    if final.get("congestion_prob", 1.0) > 0.5:
        recommendation = "Prioritize tower rebalancing, loss reduction, and bandwidth expansion before accepting more traffic."
    elif final.get("latency", 0.0) > 100.0:
        recommendation = "Keep rerouting active and review routing paths for avoidable delay."

    return f"""# AI Network Optimization Report

## Executive Summary

The optimization simulation ran {len(simulation_log)} steps and produced a total reward of {total_reward:.2f}. The recommended action path was: {", ".join(action_sequence) if action_sequence else "No actions"}.

## Initial vs Final Network State

{state_table}

## Action Sequence Taken

{chr(10).join(f"{index + 1}. {action}" for index, action in enumerate(action_sequence))}

## Strategy Comparison

{_strategy_table(comparison_dict)}

## Top Contributing SHAP Features

{feature_lines}

## Recommendations

{recommendation}
"""
