from openenv.core.env_server.types import Action, Observation
from pydantic import Field, BaseModel
from typing import List, Tuple, Optional


class ChaosEngineAction(Action):
    """Actions for traffic control"""

    action_type: str = Field(
        ..., description="move_up | move_down | move_left | move_right | wait"
    )


class ChaosEngineObservation(Observation):
    """State of the traffic environment"""

    grid: List[List[int]] = Field(
        ..., description="10x10 grid (0 empty, 1 vehicle, 2 EV, 3 blocked)"
    )
    ev_position: Tuple[int, int] = Field(
        ..., description="Emergency vehicle position"
    )
    ev_destination: Tuple[int, int] = Field(
        ..., description="Emergency vehicle destination"
    )
    traffic_density: float = Field(
        ..., description="Current traffic density (0–1)"
    )
    timestep: int = Field(
        ..., description="Current timestep"
    )
    max_steps: int = Field(
        ..., description="Max steps per episode"
    )

    summary: str = Field(..., description="Natural language summary")
    goal: str = Field(..., description="Task goal")
    distance_to_goal: int = Field(..., description="Distance to destination")


class ChaosEngineReward(BaseModel):
    """Reward signal"""

    value: float
    reason: str = ""