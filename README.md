# Chaos Engine 🚦

Chaos Engine is a real-world traffic control simulation environment
built using OpenEnv.\
The goal is to train and evaluate AI agents that can create a **green
corridor** for emergency vehicles while minimizing disruption to
civilian traffic.

---

## 🚨 Problem Motivation

In real cities, ambulances, fire trucks and emergency vehicles lose critical time due
to traffic congestion.

Chaos Engine simulates: - Urban traffic city grid (10x10) - Civilian
congestion - Road block incidents - Emergency vehicle routing

---

## 🧠 Environment Overview

### Observation Space

{ "city grid": "10x10 grid (0 empty, 1 vehicle, 2 EV, 3 blocked)",
"ev_position": \[x, y\], "ev_destination": \[x, y\], "traffic_density":
float, "timestep": int, "max_steps": int }

---

### Action Space

{ "action_type": "signal \| reroute \| noop", "target_id": "optional",
"value": "optional" }

---

### Reward Design

- +progress toward destination
- +100 on success
- -traffic congestion penalty
- -gridlock penalty

---

## 🎯 Tasks

1.  green_corridor_easy --- Low traffic\
2.  congestion_control_medium --- Medium traffic\
3.  incident_response_hard --- High traffic with dynamic incidents

---

## 🎥 Demo

The Chaos Engine demonstrates real-time emergency traffic coordination.

Key behaviors:
- Creates a green corridor dynamically
- Avoids congestion hotspots
- Adapts to unexpected roadblocks

This environment simulates real-world urban traffic control systems used in smart cities.

---

## ⚙️ Setup

uv sync

---

## 🐳 Docker

docker build -t chaos_engine . docker run -p 8000:8000 chaos_engine

---

## 🤖 Inference

python inference.py

Environment variables: API_BASE_URL, MODEL_NAME, HF_TOKEN

---

## 📊 Baseline Performance

Task Score

---

Easy High
Medium Moderate
Hard Challenging

---

## 🏆 Features

- Real-world traffic simulation
- Dynamic incidents
- Dense reward shaping
- OpenEnv compliant
- Dockerized server
