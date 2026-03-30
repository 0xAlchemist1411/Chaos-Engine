import os
import re
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME   = os.getenv("MODEL_NAME")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
BASE_URL     = "http://localhost:8000"

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

VALID_ACTIONS = {"move_up", "move_down", "move_left", "move_right"}

# ---------------- SYSTEM PROMPT ----------------

SYSTEM_PROMPT = """You are an intelligent RL agent controlling an emergency vehicle.

                Goal: Reach destination FAST.

                Avoid:
                - Blocks (#)
                - Traffic (C)

                Learn from previous steps.
                Return ONLY JSON:
                {"reasoning": "...", "action": "move_down"}
            """

# ---------------- GRID RENDER ----------------

def render_grid(obs):
    grid = obs["grid"]
    ev   = tuple(obs["ev_position"])
    dest = tuple(obs["ev_destination"])

    symbols = {0: ".", 1: "C", 2: "E", 3: "#"}
    rows = []

    for r in range(len(grid)):
        row = []
        for c in range(len(grid)):
            if (r, c) == ev:
                row.append("E")
            elif (r, c) == dest:
                row.append("D")
            else:
                row.append(symbols.get(grid[r][c], "?"))
        rows.append(" ".join(row))

    return "\n".join(rows)

# ---------------- PROMPT ----------------

def build_prompt(obs, history, strict=False):

    hist = "\n".join([
        f"Step {h['step']}: {h['action']} -> {h['outcome']}"
        for h in history[-5:]
    ]) if history else "None"

    strict_msg = ""
    if strict:
        strict_msg = "\nONLY JSON OUTPUT ALLOWED."

    return f"""
        GRID:
        {render_grid(obs)}

        STATE:
        EV: {obs["ev_position"]}
        DEST: {obs["ev_destination"]}
        DISTANCE: {obs["distance_to_goal"]}
        DENSITY: {obs["traffic_density"]}

        HISTORY:
        {hist}

        {strict_msg}

        Return JSON:
        {{"reasoning": "...", "action": "move_up/down/left/right"}}
        """

# ---------------- PARSE ----------------

def parse_action(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL).strip()

    try:
        data = json.loads(text)
        action = data.get("action", "").strip().lower()
        if action in VALID_ACTIONS:
            return action
    except:
        pass

    for a in VALID_ACTIONS:
        if a in text:
            return a

    return None

# ---------------- LLM ----------------

def ask_llm(obs, history):
    for i in range(3):
        strict = (i > 0)

        try:
            res = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(obs, history, strict)},
                ],
                temperature=0.2,
            )

            raw = res.choices[0].message.content.strip()
            action = parse_action(raw)

            if action:
                return action

        except Exception as e:
            print("LLM ERROR:", e)

    return "wait"  # minimal fallback

# ---------------- ACTION ----------------

def get_action(obs, history):
    action = ask_llm(obs, history)
    return {"action_type": action}

# ---------------- GRADING ----------------

def grade(task_id, obs, steps):
    dist = obs.get("distance_to_goal", 20)

    if task_id == "green_corridor_easy":
        return 1.0 if dist == 0 else max(0, 1 - dist / 20)

    elif task_id == "congestion_control_medium":
        return max(0, 1 - dist / 25)

    elif task_id == "incident_response_hard":
        score = max(0, 1 - dist / 30)

        if steps < 30:
            score += 0.1

        if obs.get("traffic_density", 1) < 0.5:
            score += 0.1

        return min(score, 1.0)

# ---------------- RUN ----------------

def run_task(task_id):

    resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id},
    ).json()

    if "observation" not in resp:
        raise Exception(f"Reset failed: {resp}")

    obs = resp["observation"]

    history = []
    total_reward = 0

    for step in range(50):

        action = get_action(obs, history)

        response = requests.post(
            f"{BASE_URL}/step",
            json={"action": action},
        )

        try:
            res = response.json()
        except:
            print("ERROR:", response.text)
            break

        if "observation" not in res:
            print("INVALID:", res)
            break

        new_obs = res["observation"]

        reward = res.get("reward", 0)
        if isinstance(reward, dict):
            reward = reward.get("value", 0)

        total_reward += reward

        outcome = "improved" if new_obs["distance_to_goal"] < obs["distance_to_goal"] else "worse"

        history.append({
            "step": step,
            "action": action["action_type"],
            "outcome": outcome
        })

        obs = new_obs

        if res.get("done", False):
            break

    return grade(task_id, obs, step + 1)

# ---------------- MAIN ----------------

if __name__ == "__main__":
    tasks = [
        "green_corridor_easy",
        "congestion_control_medium",
        "incident_response_hard",
    ]

    results = {}

    for t in tasks:
        results[t] = run_task(t)

    print("\nFINAL SCORES:")
    for k, v in results.items():
        print(f"{k}: {v:.3f}")

    print("Average:", sum(results.values()) / len(results))