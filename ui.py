"""
Chaos Engine — Interactive Gradio UI
Visualize and control the traffic simulation from your browser.
Uses the environment directly (no HTTP) for reliable state management.
"""

import os
import time
import gradio as gr
from dotenv import load_dotenv
from server.chaos_engine_environment import ChaosEngineEnvironment

load_dotenv()

# ───────────────────── Grid Rendering ─────────────────────

CELL_STYLES = {
    0: ("⬜", "#1a1a2e"),           # empty — dark
    1: ("🚗", "#e94560"),           # civilian car — red
    2: ("🚑", "#0f3460"),           # emergency vehicle — blue
    3: ("🚧", "#533483"),           # roadblock — purple
}

DEST_EMOJI = "🏥"
EV_EMOJI   = "🚑"


def render_grid_html(grid, ev_pos, destination):
    """Render 10×10 grid as a styled HTML table."""
    rows_html = ""
    for r in range(len(grid)):
        cells = ""
        for c in range(len(grid[r])):
            val = grid[r][c]
            emoji, bg = CELL_STYLES.get(val, ("❓", "#333"))

            # Override for special positions
            if [r, c] == list(ev_pos) or (r, c) == tuple(ev_pos):
                emoji = EV_EMOJI
                bg = "#00d2ff"
            elif [r, c] == list(destination) or (r, c) == tuple(destination):
                emoji = DEST_EMOJI
                bg = "#00ff88"

            cells += f"""
                <td style="
                    width:48px; height:48px;
                    text-align:center; vertical-align:middle;
                    font-size:24px;
                    background:{bg};
                    border:1px solid #2a2a4a;
                    border-radius:6px;
                    transition: all 0.3s ease;
                ">{emoji}</td>
            """
        rows_html += f"<tr>{cells}</tr>"

    return f"""
    <div style="display:flex; justify-content:center; padding:16px;">
        <table style="
            border-collapse:separate;
            border-spacing:3px;
            background:#0d0d1a;
            padding:12px;
            border-radius:16px;
            box-shadow: 0 0 40px rgba(0,210,255,0.15), 0 0 80px rgba(0,210,255,0.05);
        ">
            {rows_html}
        </table>
    </div>
    """


def render_legend():
    return """
    <div style="display:flex; gap:20px; justify-content:center; flex-wrap:wrap; padding:8px 0;">
        <span>🚑 <b style="color:#00d2ff;">Emergency Vehicle</b></span>
        <span>🏥 <b style="color:#00ff88;">Destination</b></span>
        <span>🚗 <b style="color:#e94560;">Civilian Car</b></span>
        <span>🚧 <b style="color:#533483;">Roadblock</b></span>
        <span>⬜ <b style="color:#666;">Empty</b></span>
    </div>
    """


# ───────────────────── Direct Environment Control ─────────────────────
# We instantiate the environment directly — no HTTP, no session issues.

env = ChaosEngineEnvironment()
current_obs = None
step_count = 0
rewards_list = []
is_done = False


def obs_to_dict(obs):
    """Convert a ChaosEngineObservation to a plain dict."""
    return {
        "grid": obs.grid,
        "ev_position": list(obs.ev_position),
        "ev_destination": list(obs.ev_destination),
        "traffic_density": obs.traffic_density,
        "timestep": obs.timestep,
        "max_steps": obs.max_steps,
        "distance_to_goal": obs.distance_to_goal,
        "summary": obs.summary,
        "goal": obs.goal,
        "done": obs.done,
        "reward": obs.reward,
    }


def format_status(obs, steps, total_rewards, done):
    """Build a status info HTML block."""
    if obs is None:
        return "<p style='color:#888;'>No active simulation. Select a task and click <b>Reset</b>.</p>"

    ev = obs["ev_position"]
    dest = obs["ev_destination"]
    dist = obs["distance_to_goal"]
    density = obs["traffic_density"]
    total_reward = sum(total_rewards) if total_rewards else 0
    status_emoji = "✅ Reached!" if done and dist == 0 else ("⏱️ Running" if not done else "❌ Out of steps")

    return f"""
    <div style="
        display:grid; grid-template-columns:1fr 1fr;
        gap:12px; padding:12px;
        background:#111128; border-radius:12px;
        border:1px solid #2a2a4a;
    ">
        <div><span style="color:#888;">Status</span><br><b style="font-size:18px;">{status_emoji}</b></div>
        <div><span style="color:#888;">Step</span><br><b style="font-size:18px;">{steps} / {obs.get('max_steps', 50)}</b></div>
        <div><span style="color:#888;">EV Position</span><br><b style="color:#00d2ff;">({ev[0]}, {ev[1]})</b></div>
        <div><span style="color:#888;">Destination</span><br><b style="color:#00ff88;">({dest[0]}, {dest[1]})</b></div>
        <div><span style="color:#888;">Distance</span><br><b style="font-size:18px;">{dist}</b></div>
        <div><span style="color:#888;">Density</span><br><b>{density:.1%}</b></div>
        <div style="grid-column:span 2;">
            <span style="color:#888;">Total Reward</span><br>
            <b style="font-size:20px; color:{'#00ff88' if total_reward > 0 else '#e94560'};">{total_reward:.2f}</b>
        </div>
    </div>
    """


# ───────────────────── Actions ─────────────────────

def reset_env(task_id):
    """Reset the environment with selected task."""
    global env, current_obs, step_count, rewards_list, is_done

    env = ChaosEngineEnvironment()
    step_count = 0
    rewards_list = []
    is_done = False

    try:
        obs_raw = env.reset(task_id=task_id)
        current_obs = obs_to_dict(obs_raw)
    except Exception as e:
        return (
            f"<p style='color:#e94560;'>❌ Reset error: {e}</p>",
            format_status(None, 0, [], False),
            render_legend(),
        )

    grid_html = render_grid_html(
        current_obs["grid"],
        current_obs["ev_position"],
        current_obs["ev_destination"],
    )
    status_html = format_status(current_obs, 0, [], False)
    return grid_html, status_html, render_legend()


def take_step(action_type):
    """Take a single step with the given action."""
    global current_obs, step_count, rewards_list, is_done

    if current_obs is None:
        return (
            "<p style='color:#e94560;'>⚠️ Please reset the environment first.</p>",
            format_status(None, 0, [], False),
        )
    if is_done:
        return (
            render_grid_html(current_obs["grid"], current_obs["ev_position"], current_obs["ev_destination"]),
            format_status(current_obs, step_count, rewards_list, True),
        )

    try:
        from models import ChaosEngineAction
        action = ChaosEngineAction(action_type=action_type)
        obs_raw = env.step(action)
        new_obs = obs_to_dict(obs_raw)
    except Exception as e:
        return (
            f"<p style='color:#e94560;'>❌ Step error: {e}</p>",
            format_status(current_obs, step_count, rewards_list, is_done),
        )

    reward = new_obs["reward"]
    rewards_list.append(reward)
    step_count += 1
    is_done = new_obs["done"]
    current_obs = new_obs

    grid_html = render_grid_html(
        current_obs["grid"],
        current_obs["ev_position"],
        current_obs["ev_destination"],
    )
    status_html = format_status(current_obs, step_count, rewards_list, is_done)
    return grid_html, status_html


def auto_run(task_id):
    """Run the LLM agent on the selected task (requires HF_TOKEN)."""
    global current_obs, step_count, rewards_list, is_done, env

    # Reset first
    grid_html, status_html, legend_html = reset_env(task_id)
    yield grid_html, status_html, legend_html, "<p style='color:#00d2ff;'>🤖 Starting AI agent...</p>"

    if current_obs is None:
        yield grid_html, status_html, legend_html, "<p style='color:#e94560;'>❌ Failed to reset.</p>"
        return

    # Import inference components
    try:
        from inference import get_action
    except ImportError as e:
        yield grid_html, status_html, legend_html, f"<p style='color:#e94560;'>❌ Cannot import inference: {e}. Check HF_TOKEN and dependencies.</p>"
        return

    history = []
    obs = current_obs

    for step in range(1, 51):
        if is_done:
            break

        try:
            action = get_action(obs, history)
            action_str = action["action_type"]
        except Exception as e:
            yield (
                render_grid_html(obs["grid"], obs["ev_position"], obs["ev_destination"]),
                format_status(obs, step_count, rewards_list, is_done),
                legend_html,
                f"<p style='color:#e94560;'>❌ LLM error at step {step}: {e}</p>",
            )
            return

        # Take step via direct env call
        grid_html, status_html = take_step(action_str)

        obs = current_obs
        outcome = "improved" if obs["distance_to_goal"] < (history[-1].get("dist", 999) if history else 999) else "no change"
        history.append({"step": step, "action": action_str, "outcome": outcome, "dist": obs["distance_to_goal"]})

        agent_log = f"<p style='color:#00d2ff;'>🤖 Step {step}: <b>{action_str}</b> → reward {rewards_list[-1]:.2f} | dist {obs['distance_to_goal']} {'✅ DONE!' if is_done else ''}</p>"

        yield grid_html, status_html, legend_html, agent_log
        time.sleep(0.5)

    # Final summary
    dist = obs.get("distance_to_goal", -1)
    final_msg = "🎉 Destination reached!" if dist == 0 else f"⏱️ Episode ended. Distance remaining: {dist}"
    total = sum(rewards_list)
    yield (
        render_grid_html(obs["grid"], obs["ev_position"], obs["ev_destination"]),
        format_status(obs, step_count, rewards_list, True),
        legend_html,
        f"""
        <div style="padding:16px; background:#111128; border-radius:12px; border:1px solid #2a2a4a; margin-top:8px;">
            <h3 style="color:#00d2ff; margin:0 0 8px;">{final_msg}</h3>
            <p>Total steps: <b>{step_count}</b> | Total reward: <b style="color:{'#00ff88' if total > 0 else '#e94560'};">{total:.2f}</b></p>
        </div>
        """,
    )


# ───────────────────── Custom CSS ─────────────────────

CUSTOM_CSS = """
/* Dark premium theme */
.gradio-container {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%) !important;
    font-family: 'Inter', 'Segoe UI', sans-serif !important;
}

#header-block {
    text-align: center;
    padding: 24px 16px 8px;
}

.action-btn {
    min-width: 64px !important;
    min-height: 64px !important;
    font-size: 28px !important;
    border-radius: 12px !important;
    border: 1px solid #2a2a4a !important;
    background: #111128 !important;
    transition: all 0.2s ease !important;
}
.action-btn:hover {
    background: #1a1a3e !important;
    box-shadow: 0 0 16px rgba(0,210,255,0.3) !important;
    transform: scale(1.05);
}

.primary-btn {
    background: linear-gradient(135deg, #00d2ff, #3a7bd5) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
    transition: all 0.2s ease !important;
}
.primary-btn:hover {
    box-shadow: 0 0 20px rgba(0,210,255,0.4) !important;
    transform: translateY(-1px);
}

.reset-btn {
    background: linear-gradient(135deg, #e94560, #c23152) !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 10px !important;
}

.info-card {
    background: #111128 !important;
    border: 1px solid #2a2a4a !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

#footer-text {
    text-align: center;
    padding: 16px;
    opacity: 0.5;
}
"""


# ───────────────────── Gradio App ─────────────────────

def build_ui():
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="Chaos Engine 🚦 — Traffic Control Simulation",
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.gray,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="#0d0d1a",
            body_text_color="#e0e0e0",
            block_background_fill="#111128",
            block_border_color="#2a2a4a",
            block_label_text_color="#888",
            button_primary_background_fill="linear-gradient(135deg, #00d2ff, #3a7bd5)",
            button_primary_text_color="white",
            input_background_fill="#1a1a2e",
            input_border_color="#2a2a4a",
        ),
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="header-block">
            <h1 style="
                font-size:42px; margin:0;
                background: linear-gradient(135deg, #00d2ff, #3a7bd5, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 800;
            ">🚦 Chaos Engine</h1>
            <p style="color:#888; font-size:16px; margin:8px 0 0;">
                Real-world Traffic Control Simulation for Emergency Vehicle Routing
            </p>
            <p style="color:#555; font-size:13px; margin:4px 0 0;">
                Navigate the emergency vehicle 🚑 to the hospital 🏥 while avoiding traffic 🚗 and roadblocks 🚧
            </p>
        </div>
        """)

        with gr.Row():
            # ── Left Panel : Controls ──
            with gr.Column(scale=1, min_width=280):
                gr.HTML("<h3 style='color:#00d2ff; margin:0 0 8px;'>⚙️ Controls</h3>")

                task_dropdown = gr.Dropdown(
                    choices=[
                        ("🟢 Green Corridor — Easy", "green_corridor_easy"),
                        ("🟡 Congestion Control — Medium", "congestion_control_medium"),
                        ("🔴 Incident Response — Hard", "incident_response_hard"),
                    ],
                    value="green_corridor_easy",
                    label="Select Task",
                    interactive=True,
                )

                with gr.Row():
                    reset_btn = gr.Button("🔄 Reset", elem_classes=["reset-btn"], size="lg")
                    auto_btn = gr.Button("🤖 Auto Run (AI)", elem_classes=["primary-btn"], size="lg")

                gr.HTML("<h3 style='color:#00d2ff; margin:16px 0 8px;'>🎮 Manual Controls</h3>")

                with gr.Column():
                    with gr.Row():
                        gr.HTML("<span></span>")
                        up_btn = gr.Button("⬆️", elem_classes=["action-btn"])
                        gr.HTML("<span></span>")
                    with gr.Row():
                        left_btn = gr.Button("⬅️", elem_classes=["action-btn"])
                        down_btn = gr.Button("⬇️", elem_classes=["action-btn"])
                        right_btn = gr.Button("➡️", elem_classes=["action-btn"])

                gr.HTML("<h3 style='color:#00d2ff; margin:16px 0 8px;'>📊 Status</h3>")
                status_display = gr.HTML(
                    value=format_status(None, 0, [], False),
                    elem_classes=["info-card"],
                )

            # ── Right Panel : Grid ──
            with gr.Column(scale=2, min_width=500):
                legend_display = gr.HTML(value=render_legend())
                grid_display = gr.HTML(
                    value="""
                    <div style="
                        display:flex; align-items:center; justify-content:center;
                        min-height:400px; color:#444; font-size:18px;
                    ">
                        Select a task and click <b style="color:#e94560; margin:0 6px;">Reset</b> to begin
                    </div>
                    """,
                )
                agent_log = gr.HTML(value="", visible=True)

        # ── Footer ──
        gr.HTML("""
        <div id="footer-text">
            <p style="font-size:12px; color:#444;">
                Chaos Engine — Built for the
                <a href="https://huggingface.co/spaces/open-env/benchmark" target="_blank" style="color:#3a7bd5;">OpenEnv Benchmark</a>
                 | <a href="https://github.com/0xAlchemist1411/Chaos-Engine" target="_blank" style="color:#3a7bd5;">GitHub</a>
            </p>
        </div>
        """)

        # ── Events ──
        reset_btn.click(
            fn=reset_env,
            inputs=[task_dropdown],
            outputs=[grid_display, status_display, legend_display],
        )

        up_btn.click(
            fn=lambda: take_step("move_up"),
            outputs=[grid_display, status_display],
        )
        down_btn.click(
            fn=lambda: take_step("move_down"),
            outputs=[grid_display, status_display],
        )
        left_btn.click(
            fn=lambda: take_step("move_left"),
            outputs=[grid_display, status_display],
        )
        right_btn.click(
            fn=lambda: take_step("move_right"),
            outputs=[grid_display, status_display],
        )

        auto_btn.click(
            fn=auto_run,
            inputs=[task_dropdown],
            outputs=[grid_display, status_display, legend_display, agent_log],
        )

    return demo


# ───────────────────── Entry Point ─────────────────────

if __name__ == "__main__":
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
