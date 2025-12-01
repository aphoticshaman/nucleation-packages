"""Outcome Gradient Steering Engine (OGSE) - The active influence layer."""

from .outcome_steering import (
    SteeringObjective,
    SteeringAction,
    OutcomeGradientSteeringEngine,
    compute_outcome_gradient,
    project_to_action_space,
    evaluate_steering_effectiveness,
)

__all__ = [
    "SteeringObjective",
    "SteeringAction",
    "OutcomeGradientSteeringEngine",
    "compute_outcome_gradient",
    "project_to_action_space",
    "evaluate_steering_effectiveness",
]
