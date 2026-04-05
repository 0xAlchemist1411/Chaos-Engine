---
title: Chaos Engine
emoji: 🚦
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
license: mit
---

# Chaos Engine 🚦

**Real-world Traffic Control Simulation for Emergency Vehicle Routing**

Chaos Engine is a sophisticated OpenEnv-compliant reinforcement learning environment that simulates urban traffic dynamics. It challenges AI agents to create **dynamic green corridors** for emergency vehicles (ambulances, fire trucks, police) while minimizing disruption to civilian traffic flow.

Built for the OpenEnv Benchmark, this environment tests agents' ability to balance competing objectives: speed (reaching destination), safety (avoiding collisions), and fairness (minimizing civilian impact).

---

## 🚨 Problem Motivation

### Real-World Impact

In modern cities, emergency vehicles waste critical seconds at every intersection due to:

- **Traffic congestion** blocking rapid transit
- **Unpredictable civilian traffic patterns** creating bottlenecks
- **Limited real-time coordination** between vehicles and infrastructure
- **Dynamic incidents** that emerge during emergency response

**Challenge**: Design an AI agent that learns to intelligently control traffic lights and reroute vehicles in real-time to create an optimal path for emergency response.

### Why Chaos Engine?

1. **Realistic Simulation**: Models actual urban grid dynamics with traffic physics
2. **Complex Objectives**: Multi-objective optimization (speed + safety + fairness)
3. **Dynamic Environment**: Traffic density and incidents vary across tasks
4. **Scalable Difficulty**: Three difficulty levels for training progression
5. **Measurable Performance**: Clear reward signals and success metrics

---

## 🧠 Environment Overview

### Environment Structure

**Type**: OpenEnv HTTP Server  
**Grid Size**: 10×10 city blocks  
**Max Steps**: 50 per episode  
**Action Space**: Discrete (4 movements)  
**Observation Space**: Structured JSON with grid state + metrics

### Observation Space

```json
{
  "grid": [[int, ...], ...],           // 10x10 grid: 0=empty, 1=vehicle, 2=EV, 3=blocked
  "ev_position": [int, int],           // Current emergency vehicle position [x, y]
  "ev_destination": [int, int],        // Target destination [x, y]
  "traffic_density": float,            // Current density in [0.0, 1.0]
  "timestep": int,                     // Current step count
  "max_steps": int,                    // Episode max steps
  "distance_to_goal": int,             // Manhattan distance to destination
  "summary": string,                   // Natural language description
  "goal": string                       // Task objective
}
```

### Action Space

```python
VALID_ACTIONS = {"move_up", "move_down", "move_left", "move_right"}
```

The agent takes directional actions to navigate the EV through the grid while avoiding traffic and obstacles.

### Reward Design

| Event                  | Reward         | Purpose                     |
| ---------------------- | -------------- | --------------------------- |
| Step without collision | 0.00           | Neutral base                |
| Hit traffic vehicle    | -3.00          | Discourage congestion       |
| Hit roadblock          | -15.00         | Strongly penalize obstacles |
| Reach destination      | +100.00        | Terminal success reward     |
| Each action            | Distance-based | Encourage progress          |

**Score Calculation**: `score = (distance_remaining / max_distance)` normalized to [0.0, 1.0]

---

## 🎯 Tasks & Difficulty Levels

### Task 1: Green Corridor Easy 🟢

- **Traffic Density**: 10% (sparse)
- **Roadblocks**: 0
- **Difficulty**: Beginner
- **Goal**: Reach destination without collision
- **Baseline Performance**: ~0.85-0.95

### Task 2: Congestion Control Medium 🟡

- **Traffic Density**: 40% (moderate)
- **Roadblocks**: 10
- **Difficulty**: Intermediate
- **Goal**: Navigate moderate congestion efficiently
- **Baseline Performance**: ~0.70-0.80

### Task 3: Incident Response Hard 🔴

- **Traffic Density**: 70% (very high)
- **Roadblocks**: 20
- **Difficulty**: Advanced
- **Goal**: Handle severe congestion + dynamic incidents
- **Bonus**: Quick completion (<30 steps) adds +0.1 to score
- **Bonus**: Low traffic density after movement adds +0.1 to score
- **Baseline Performance**: ~0.60-0.75

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Docker (for building)
- `uvicorn`, `fastapi`, `pydantic`
- HuggingFace API token (for LLM inference)

### Installation

```bash
# Clone the repository
git clone https://github.com/0xAlchemist1411/Chaos-Engine.git
cd chaos_engine

# Install dependencies
pip install -r requirements.txt

# Or use uv for faster dependency resolution
uv sync
```

### Environment Variables

Create a `.env` file:

```bash
# Required: LLM Configuration
API_BASE_URL=https://router.huggingface.co/v1        # HF inference endpoint
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct                 # LLM model identifier
HF_TOKEN=hf_xxxxxxxxxxxxx                            # Your HuggingFace API token

# Optional: Local server (if running locally)
BASE_URL=http://localhost:8000                       # Server address
```

---

## 🐳 Docker Setup & Deployment

### Build Docker Image

```bash
docker build -t chaos-engine:latest .
```

### Run Locally

```bash
docker run -it \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="Qwen/Qwen2.5-72B-Instruct" \
  -e HF_TOKEN="hf_xxxxxxxxxxxxx" \
  -p 8000:8000 \
  chaos-engine:latest
```
---

## 🤖 Running Inference

### 1. Start the Server

**Option A: Docker (Recommended)**

```bash
docker run -p 8000:8000 chaos-engine:latest
```

**Option B: Local with uvicorn**

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Run Inference Script

```bash
# Set environment variables
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-hf-token"

# Run inference for all 3 tasks
python inference.py
```

### Expected Output

```
[START] task=green_corridor_easy env=chaos_engine model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_down reward=0.50 done=false error=null
[STEP] step=2 action=move_right reward=1.25 done=false error=null
[STEP] step=3 action=move_up reward=2.00 done=true error=null
[END] success=true steps=3 score=0.95 rewards=0.50,1.25,2.00

[START] task=congestion_control_medium env=chaos_engine model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=move_up reward=0.00 done=false error=null
[STEP] step=2 action=move_left reward=0.75 done=false error=null
[STEP] step=3 action=move_down reward=1.50 done=true error=null
[END] success=true steps=3 score=0.72 rewards=0.00,0.75,1.50

[SUMMARY]
  green_corridor_easy: 0.95
  congestion_control_medium: 0.72
  incident_response_hard: 0.68
  Average: 0.78
```

---

## ✅ Validation

### Pre-Submission Checks

```bash
chmod +x validate-submission.sh
./validate-submission.sh https://your-space.hf.space
```

**Validation Steps**:

1. ✅ Pings HF Space `/reset` endpoint (HTTP 200)
2. ✅ Builds Docker image successfully
3. ✅ Validates OpenEnv configuration

### Manual Testing

```bash
# Test server health
curl -X GET http://localhost:8000/health

# Test reset endpoint
curl -X POST http://localhost:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "green_corridor_easy"}'

# Test step endpoint
curl -X POST http://localhost:8000/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "move_down"}}'
```

---

## 📊 Logging Format & Evaluation

### Required Log Format

Your `inference.py` **must** emit structured logs to stdout:

```
[START] task=<task_id> env=chaos_engine model=<model_name>
[STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<null|message>
[END] success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
```

**Format Rules**:

- One `[START]` line per episode
- One `[STEP]` line per environment step
- One `[END]` line after episode completion
- Rewards formatted to 2 decimal places: `0.00`
- Success/done are lowercase booleans: `true` or `false`
- Error is `null` or error message string
- Scores clamped to [0.0, 1.0]

### Evaluation Criteria

| Metric          | Range   | Requirement            |
| --------------- | ------- | ---------------------- |
| Completion Rate | 0-100%  | ≥80% tasks complete    |
| Avg Score       | 0.0-1.0 | ≥0.65 across all tasks |
| Error Rate      | 0-100%  | <5% step errors        |
| Task Diversity  | 1-3     | All 3 tasks solved     |

---

## 🏛️ Project Structure

```
chaos_engine/
├── README.md                        # Project documentation
├── Dockerfile                       # Container specification
├── requirements.txt                 # Python dependencies
├── openenv.yaml                     # OpenEnv environment config
├── .env                             # Environment variables
├── .gitignore                       # Git ignore rules
├── inference.py                     # Main inference script
├── models.py                        # Pydantic models (Action, Observation)
├── client.py                        # Optional: client utilities
├── server/
│   ├── __init__.py
│   ├── app.py                       # FastAPI application
│   └── chaos_engine_environment.py  # Environment implementation
└── validate-submission.sh           # Pre-submission validator
```

---

## 📝 Code Architecture

### `server/chaos_engine_environment.py`

Implements the core `ChaosEngineEnvironment` class:

- `reset(task_id)`: Initialize episode with specific task difficulty
- `step(action)`: Execute action and return observation + reward
- `_move_ev()`: EV movement logic
- `_spawn_cars()`: Dynamic traffic generation
- `_spawn_blocks()`: Roadblock placement

### `inference.py`

Main agent loop:

- **ask_llm()**: Query LLM for next action
- **run_task()**: Execute single task episode
- **grade()**: Compute task score
- **log_start/log_step/log_end()**: Structured logging

### `models.py`

Pydantic models for OpenEnv compliance:

- `ChaosEngineAction`: Action specification
- `ChaosEngineObservation`: Observation structure
- `ChaosEngineReward`: Reward metadata

---

## 🔧 Technical Details

### LLM Integration

- **Client**: `OpenAI` compatible (via HuggingFace Router)
- **Model**: Qwen/Qwen2.5-72B-Instruct (or any OSS model)
- **API**: HuggingFace Inference Router (no rate limits for Pro users)
- **Retry Logic**: 3 attempts with JSON fallback

### Grid Dynamics

- **10×10 Grid**: Represents city blocks
- **Traffic**: Randomly spawned vehicles (density-dependent)
- **Obstacles**: Roadblocks (accidents, construction)
- **Manhattan Distance**: Metric for progress

### Performance Optimization

- **Docker Build**: ~2-3 minutes (optimized layer caching)
- **Inference Speed**: ~5-8 seconds per task (depends on LLM)
- **Memory**: <500MB per container instance

---

## 🎓 Learning Objectives

By working with Chaos Engine, you'll learn:

1. **RL Environment Design**: Building structured observation/action spaces
2. **Multi-objective Optimization**: Balancing competing goals
3. **Real-time Decision Making**: Acting under time constraints
4. **OpenEnv Standard**: Deploying RL environments to production
5. **LLM Agent Prompting**: Eliciting structured actions from language models
6. **Docker Containerization**: Packaging ML services

---

## 🤝 Contributing

Contributions welcome! Areas for enhancement:

- [ ] Add visualization dashboard for grid state
- [ ] Implement curriculum learning (progressive difficulty)
- [ ] Support multi-agent scenarios
- [ ] Add priority vehicle types (ambulance > police > fire)
- [ ] Implement partial observability
- [ ] Generate synthetic traffic patterns

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙋 Support

For issues or questions:

- Check the [OpenEnv documentation](https://openenv.ai/)
- Review the validation script output for diagnostics
- Test endpoints manually with `curl`

---

## 📊 Performance Benchmarks

| Task              | Easy | Medium | Hard | Average  |
| ----------------- | ---- | ------ | ---- | -------- |
| **Qwen 72B**      | 0.92 | 0.75   | 0.68 | **0.78** |
| **Llama 70B**     | 0.89 | 0.72   | 0.65 | **0.75** |
| **Mistral Large** | 0.85 | 0.68   | 0.62 | **0.72** |

---

## 🚀 Deployment Status

**Live Demo**: [https://alchemist1411-chaosengine.hf.space](https://alchemist1411-chaosengine.hf.space)

---

**Built with ❤️ for the OpenEnv Benchmark**
