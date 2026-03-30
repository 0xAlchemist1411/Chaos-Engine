from openenv.core.env_server.http_server import create_app

from models import (
    ChaosEngineAction,
    ChaosEngineObservation,
    ChaosEngineReward,
)

from server.chaos_engine_environment import ChaosEngineEnvironment

app = create_app(
    ChaosEngineEnvironment,
    ChaosEngineAction,
    ChaosEngineObservation,
    env_name="chaos_engine",
    max_concurrent_envs=2,
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/render")
def render():
    env = ChaosEngineEnvironment()
    return {
        "grid": env.grid if hasattr(env, "grid") else [],
        "ev": getattr(env, "ev_pos", None),
        "destination": getattr(env, "destination", None),
    }

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()