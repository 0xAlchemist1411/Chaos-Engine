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
        self.max_steps = 30
        self.task_id = "green_corridor_easy"

        self._state = State(episode_id=str(uuid4()), step_count=0)

        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        self.ev_pos = (0, 0)
        self.destination = (self.grid_size - 1, self.grid_size - 1)

    def reset(self, task_id: str = None) -> ChaosEngineObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)

        if task_id:
            self.task_id = task_id

        self.grid = [[0 for _ in range(self.grid_size)] for _ in range(self.grid_size)]

        self.ev_pos = (0, 0)
        self.destination = (self.grid_size - 1, self.grid_size - 1)

        if self.task_id == "green_corridor_easy":
            cars, blocks = 10, 0

        elif self.task_id == "congestion_control_medium":
            cars, blocks = 40, 10

        else:  # hard
            cars, blocks = 70, 20

        self._spawn_cars(cars)
        self._spawn_blocks(blocks)

        self.grid[self.ev_pos[0]][self.ev_pos[1]] = 2

        return self._build_obs(done=False, reward=0)

    def step(self, action: ChaosEngineAction) -> ChaosEngineObservation:

        self._state.step_count += 1

        action_type = action.get("action_type") if isinstance(action, dict) else action.action_type

        prev_distance = self._distance(self.ev_pos, self.destination)

        new_pos = self._move_ev(action_type)

        reward = 0

        cell = self.grid[new_pos[0]][new_pos[1]]

        if cell == 3:
            reward -= 15

        else:
            self.grid[self.ev_pos[0]][self.ev_pos[1]] = 0

            # ⚠️ CAR
            if cell == 1:
                reward -= 3

            self.ev_pos = new_pos
            self.grid[self.ev_pos[0]][self.ev_pos[1]] = 2

        if self.task_id == "incident_response_hard":
            if random.random() < 0.2:
                self._spawn_blocks(2)

        new_distance = self._distance(self.ev_pos, self.destination)

        progress = (prev_distance - new_distance)
        reward += progress * 3

        if progress == 0:
            reward -= 5

        density = self._density()
        reward -= density * 20

        reached = self.ev_pos == self.destination

        if reached:
            reward += 200

        done = reached or self._state.step_count >= self.max_steps

        return self._build_obs(done=done, reward=reward)

    @property
    def state(self) -> State:
        return State(
            episode_id=self._state.episode_id,
            step_count=self._state.step_count,
            metadata={
                "ev_position": self.ev_pos,
                "destination": self.destination,
                "distance": self._distance(self.ev_pos, self.destination),
                "traffic_density": self._density(),
            }
        )

    # ---------------- HELPERS ----------------

    def _build_obs(self, done, reward):
        return ChaosEngineObservation(
            grid=self.grid,
            ev_position=self.ev_pos,
            ev_destination=self.destination,
            traffic_density=self._density(),
            timestep=self._state.step_count,
            max_steps=self.max_steps,
            summary=self._summary(),
            goal="Reach destination fast avoiding congestion",
            distance_to_goal=self._distance(self.ev_pos, self.destination),
            done=done,
            reward=reward,
            metadata={}
        )

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

    def _move_ev(self, action):
        x, y = self.ev_pos

        if action == "move_up":
            x -= 1
        elif action == "move_down":
            x += 1
        elif action == "move_left":
            y -= 1
        elif action == "move_right":
            y += 1

        x = max(0, min(9, x))
        y = max(0, min(9, y))

        return (x, y)

    def _summary(self):
        return f"EV at {self.ev_pos}, dest {self.destination}, density {round(self._density(), 2)}"