import os
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

BASE_URL = "http://localhost:8000"

def get_action(observation):
    prompt = f"""
        You are controlling an emergency vehicle.

        Goal: {observation.get("goal")}

        State:
        {observation.get("summary")}

        Distance to goal: {observation.get("distance_to_goal")}

        Choose ONE:
        move_up, move_down, move_left, move_right, wait

        Return ONLY action.
    """

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        action = response.choices[0].message.content.strip().lower()
    except Exception:
        action = "wait"

    if action not in ["move_up", "move_down", "move_left", "move_right"]:
        action = "wait"

    return {"action_type": action}


def run_task(task_id):
    obs = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id}
    ).json()

    total_reward = 0

    for _ in range(30):
        action = get_action(obs)

        res = requests.post(
            f"{BASE_URL}/step",
            json=action
        ).json()

        obs = res["observation"]
        total_reward += res["reward"]["value"]

        if res["done"]:
            break

    return total_reward


if __name__ == "__main__":
    tasks = [
        "green_corridor_easy",
        "congestion_control_medium",
        "incident_response_hard"
    ]

    for task in tasks:
        score = run_task(task)
        print(f"{task}: {score}")