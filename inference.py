import os
import requests
from openai import OpenAI
from dotenv import load_dotenv
from collections import deque

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
BASE_URL = os.getenv("BASE_URL") or "http://localhost:8000"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

def get_best_move(observation):
    grid = observation.get("grid")
    start = tuple(observation.get("ev_position"))
    target = tuple(observation.get("ev_destination"))

    directions = [
        ("move_up", (-1, 0)),
        ("move_down", (1, 0)),
        ("move_left", (0, -1)),
        ("move_right", (0, 1)),
    ]

    queue = deque([(start, [])])
    visited = set([start])

    while queue:
        (x, y), path = queue.popleft()

        if (x, y) == target:
            return path[0] if path else "wait"

        for action, (dx, dy) in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < 10 and 0 <= ny < 10:
                if (nx, ny) not in visited and grid[nx][ny] != 3:
                    visited.add((nx, ny))
                    queue.append(((nx, ny), path + [action]))

    return "wait"


def get_action(observation):
    bfs_action = get_best_move(observation)

    action = bfs_action

    import random
    if random.random() < 0.1:
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{
                    "role": "user",
                    "content": f"""
                        You are controlling an emergency vehicle.

                        Current state:
                        {observation}

                        Suggest best move:
                        move_up, move_down, move_left, move_right

                        Return ONLY one word.
                        """
                }],
            temperature=0.2,
        )

            llm_action = response.choices[0].message.content.strip().lower()
            llm_action = llm_action.replace(" ", "_")

            if llm_action in ["move_up", "move_down", "move_left", "move_right"]:
                action = llm_action

        except:
            pass

    if action == "wait":
        ev = observation["ev_position"]
        dest = observation["ev_destination"]

        if ev[0] < dest[0]:
            action = "move_down"
        elif ev[0] > dest[0]:
            action = "move_up"
        elif ev[1] < dest[1]:
            action = "move_right"
        elif ev[1] > dest[1]:
            action = "move_left"

    return {"action_type": action}

def grade(task_id, obs, steps):
    distance = obs.get("distance_to_goal", 20)

    if task_id == "green_corridor_easy":
        return 1.0 if distance == 0 else max(0, 1 - distance / 20)

    elif task_id == "congestion_control_medium":
        return max(0, 1 - distance / 25)

    elif task_id == "incident_response_hard":
        score = 0

        if distance == 0:
            score += 0.6

        if steps < 30:
            score += 0.2

        if obs.get("traffic_density", 1) < 0.5:
            score += 0.2

        return min(score, 1.0)


def run_task(task_id):
    obs = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id}
    ).json()

    if "observation" in obs:
        obs = obs["observation"]

    total_reward = 0

    for step in range(30):
        action = get_action(obs)

        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": action},
            headers={"Content-Type": "application/json"}
        )

        try:
            res = response.json()
        except:
            print("ERROR:", response.text)
            break

        # observation
        if "observation" in res:
            obs = res["observation"]
        else:
            obs = res

        # reward
        reward = res.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("value", 0)

        total_reward += reward

        # done
        done = res.get("done", False)
        if not done and "observation" in res:
            done = res["observation"].get("done", False)

        if done:
            break

    return grade(task_id, obs, step + 1)

if __name__ == "__main__":
    tasks = [
        "green_corridor_easy",
        "congestion_control_medium",
        "incident_response_hard"
    ]

    for task in tasks:
        score = run_task(task)
        print(f"{task}: {score}")