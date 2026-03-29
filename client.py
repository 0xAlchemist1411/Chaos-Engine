from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    ChaosEngineAction,
    ChaosEngineObservation,
    ChaosEngineReward,
)


class ChaosEngineEnv(
    EnvClient[ChaosEngineAction, ChaosEngineObservation, State]
):
    """
    Client for Chaos Engine (traffic control environment)
    """

    def _step_payload(self, action: ChaosEngineAction) -> Dict:
        return {
            "action_type": action.action_type,
            "target_id": action.target_id,
            "value": action.value,
        }

    def _parse_result(self, payload: Dict) -> StepResult[ChaosEngineObservation]:
        obs_data = payload.get("observation", {})

        observation = ChaosEngineObservation(
            grid=obs_data.get("grid", []),
            ev_position=tuple(obs_data.get("ev_position", (0, 0))),
            ev_destination=tuple(obs_data.get("ev_destination", (0, 0))),
            traffic_density=obs_data.get("traffic_density", 0.0),
            timestep=obs_data.get("timestep", 0),
            max_steps=obs_data.get("max_steps", 0),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        reward_data = payload.get("reward", {})

        reward = ChaosEngineReward(
            value=reward_data.get("value", 0.0),
            reason=reward_data.get("reason", ""),
        )

        return StepResult(
            observation=observation,
            reward=reward.value,
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )