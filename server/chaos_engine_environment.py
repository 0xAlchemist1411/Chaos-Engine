from uuid import uuid4
import random

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import (
    ChaosEngineAction,
    ChaosEngineObservation,
)

class ChaosEngineEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        random.seed(42)

        self.grid_size = 10
        self.max_steps = 50
        self.task_id = "green_corridor_easy"

        self._state = State(episode_id=str(uuid4()), step_count=0)

    def reset(self) -> ChaosEngineObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        # reset grid properly
        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        # reset positions
        self.ev_pos = (0, 0)
        self.destination = (9, 9)

        # difficulty setup
        if self.task_id == "green_corridor_easy":
            cars, blocks = 20, 0
        elif self.task_id == "congestion_control_medium":
            cars, blocks = 50, 5
        else:
            cars, blocks = 70, 8

        self._spawn_cars(cars)
        self._spawn_blocks(blocks)

        # mark EV AFTER spawning
        self.grid[self.ev_pos[0]][self.ev_pos[1]] = 2

        return ChaosEngineObservation(
            grid=self.grid,
            ev_position=self.ev_pos,
            ev_destination=self.destination,
            traffic_density=self._density(),
            timestep=0,
            max_steps=self.max_steps,
            summary=self._summary(),
            goal="Move the emergency vehicle to destination as fast as possible",
            distance_to_goal=self._distance(self.ev_pos, self.destination),
            done=False,
            metadata={},
        )

    def step(self, action: ChaosEngineAction) -> ChaosEngineObservation:

        if not hasattr(self, "ev_pos"):
            self.reset()

        self._state.step_count += 1

        if isinstance(action, dict):
            action_type = action.get("action_type", "wait")
        else:
            action_type = action.action_type

        prev_distance = self._distance(self.ev_pos, self.destination)

        # compute new position
        new_pos = self._move_ev(action_type)

        reward_value = 0

        # block collision
        if self.grid[new_pos[0]][new_pos[1]] == 3:
            reward_value -= 10

        # car collision (treat as obstacle)
        elif self.grid[new_pos[0]][new_pos[1]] == 1:
            reward_value -= 5

        else:
            # clear old EV position
            self.grid[self.ev_pos[0]][self.ev_pos[1]] = 0

            # move EV
            self.ev_pos = new_pos

            # mark new position
            self.grid[self.ev_pos[0]][self.ev_pos[1]] = 2

        # dynamic difficulty
        if self.task_id == "incident_response_hard":
            if random.random() < 0.1:
                self._spawn_blocks(1)

        new_distance = self._distance(self.ev_pos, self.destination)

        # discourage useless moves
        if prev_distance == new_distance:
            reward_value -= 2

        # reward shaping
        reward_value += (prev_distance - new_distance)

        reached = self.ev_pos == self.destination
        if reached:
            reward_value += 100

        density = self._density()
        reward_value -= density * 10

        done = reached or self._state.step_count >= self.max_steps

        return ChaosEngineObservation(
            grid=self.grid,
            ev_position=self.ev_pos,
            ev_destination=self.destination,
            traffic_density=density,
            timestep=self._state.step_count,
            max_steps=self.max_steps,
            summary=self._summary(),
            goal="Move the emergency vehicle to destination as fast as possible",
            distance_to_goal=self._distance(self.ev_pos, self.destination),
            done=done,
            reward=reward_value,
            metadata={
                "distance": self._distance(self.ev_pos, self.destination),
                "density": density
            },
        )

    @property
    def state(self) -> State:
        return self._state

    # ---------------- helpers ----------------

    def _spawn_cars(self, count):
        for _ in range(count):
            x, y = random.randint(0, 9), random.randint(0, 9)

            if (x, y) != self.ev_pos and (x, y) != self.destination:
                self.grid[x][y] = 1

    def _spawn_blocks(self, count):
        for _ in range(count):
            x, y = random.randint(0, 9), random.randint(0, 9)

            if (x, y) != self.ev_pos and (x, y) != self.destination:
                self.grid[x][y] = 3

    def _density(self):
        total = sum(cell == 1 for row in self.grid for cell in row)
        return total / (self.grid_size * self.grid_size)

    def _distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _move_ev(self, action_type):
        x, y = self.ev_pos

        if action_type == "move_up":
            x -= 1
        elif action_type == "move_down":
            x += 1
        elif action_type == "move_left":
            y -= 1
        elif action_type == "move_right":
            y += 1

        x = max(0, min(9, x))
        y = max(0, min(9, y))

        return (x, y)

    def _summary(self):
        return f"EV at {self.ev_pos}, destination {self.destination}, density {round(self._density(), 2)}"