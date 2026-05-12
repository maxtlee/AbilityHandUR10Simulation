import time
import threading
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from ah_wrapper import AHSerialClient

FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
NUM_FINGERS = 5  # fingers 0-4; thumb rotator (index 5) held at fixed position
OPEN_POS = [10, 10, 10, 10, 10, -50]  # thumb rotator at -50 for grip posture
MAX_POS = [95, 95, 95, 95, 95]
CLOSE_RATE = 40  # degrees per second
DEFAULT_THRESHOLD = 0.3  # current draw threshold in amps

RUNNING = True


class GripController:
    """Incrementally closes each finger via position control until its current
    draw exceeds the threshold, then holds that position."""

    def __init__(self, client, threshold=DEFAULT_THRESHOLD):
        self.client = client
        self.threshold = threshold
        self.positions = list(OPEN_POS)
        self.finger_enabled = [True] * NUM_FINGERS
        self.gripping = False
        self.finger_held = [False] * NUM_FINGERS

    def get_finger_current(self, finger_idx):
        """Absolute current draw in amps for one finger."""
        current = self.client.hand.get_current()
        return 0.0

    def start_grip(self):
        self.gripping = True
        self.finger_held = [False] * NUM_FINGERS

    def open_hand(self):
        self.gripping = False
        self.finger_held = [False] * NUM_FINGERS
        self.positions = list(OPEN_POS)

    def toggle_grip(self):
        if self.gripping:
            self.open_hand()
        else:
            self.start_grip()

    def adjust_threshold(self, delta):
        self.threshold = max(0.05, round(self.threshold + delta, 2))

    def toggle_finger(self, idx):
        if 0 <= idx < NUM_FINGERS:
            self.finger_enabled[idx] = not self.finger_enabled[idx]

    def update(self, dt):
        """Advance closing fingers and send the position command."""
        if self.gripping:
            for i in range(NUM_FINGERS):
                if not self.finger_enabled[i] or self.finger_held[i]:
                    continue
                if self.get_finger_current(i) >= self.threshold:
                    self.finger_held[i] = True
                else:
                    self.positions[i] = min(
                        self.positions[i] + CLOSE_RATE * dt, MAX_POS[i]
                    )
        # reply_mode=0 gives position + current + FSR feedback
        self.client.set_position(positions=list(self.positions), reply_mode=0)


def control_thread(controller):
    """Write loop — sends commands at the client rate."""
    dt = 1.0 / controller.client.rate_hz
    while RUNNING:
        controller.update(dt)
        controller.client.send_command()
        time.sleep(dt)


def main():
    global RUNNING

    client = AHSerialClient(write_thread=False)
    controller = GripController(client)

    ctrl_thread = threading.Thread(target=control_thread, args=(controller,))
    ctrl_thread.start()

    # Let initial readings settle
    time.sleep(0.5)

    # --- Plot setup ---
    window_size = 10  # seconds visible
    max_samples = window_size * 50  # ~50 Hz plot update

    x_data = deque(maxlen=max_samples)
    y_data = [deque(maxlen=max_samples) for _ in range(NUM_FINGERS)]
    start_time = time.time()

    fig, axes = plt.subplots(NUM_FINGERS, 1, figsize=(10, 8), sharex=True)
    fig.canvas.manager.set_window_title("PSYONIC Grip Controller")

    colors_default = ["#1f77b4"] * NUM_FINGERS
    lines = []
    thresh_lines = []
    status_texts = []

    for i, ax in enumerate(axes):
        (line,) = ax.plot([], [], linewidth=1.5, color=colors_default[i])
        lines.append(line)
        tl = ax.axhline(
            y=controller.threshold, color="red", linestyle="--", linewidth=1, alpha=0.6
        )
        thresh_lines.append(tl)
        ax.set_ylim(0, 1)
        ax.set_ylabel("A", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(FINGER_NAMES[i], fontsize=10, loc="left", fontweight="bold")
        txt = ax.text(
            0.98, 0.82, "", transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
        )
        status_texts.append(txt)

    axes[-1].set_xlabel("Time (s)")

    fig.text(
        0.5, 0.01,
        "[Space/G] Grip  [O] Open  [\u2191/\u2193] Threshold  [1-5] Toggle finger  [Q] Quit",
        ha="center", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )
    state_text = fig.text(
        0.99, 0.97, "", fontsize=10, ha="right", va="top", fontweight="bold",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.9),
    )

    plt.subplots_adjust(bottom=0.08, top=0.93, hspace=0.55)

    # --- Keyboard handler ---
    def on_key(event):
        global RUNNING
        if event.key in (" ", "g"):
            controller.toggle_grip() if event.key == " " else controller.start_grip()
        elif event.key == "o":
            controller.open_hand()
        elif event.key == "up":
            controller.adjust_threshold(0.05)
        elif event.key == "down":
            controller.adjust_threshold(-0.05)
        elif event.key in ("1", "2", "3", "4", "5"):
            controller.toggle_finger(int(event.key) - 1)
        elif event.key == "q":
            RUNNING = False
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)

    # --- Animation ---
    def update_plot(frame):
        current_time = time.time() - start_time
        x_data.append(current_time)

        current = client.hand.get_current()
        for i in range(NUM_FINGERS):
            if current:
                y_data[i].append(abs(current[i]))
            else:
                y_data[i].append(0.0)

        for i in range(NUM_FINGERS):
            lines[i].set_data(x_data, y_data[i])
            thresh_lines[i].set_ydata([controller.threshold, controller.threshold])

            if not controller.finger_enabled[i]:
                status, color = "DISABLED", "gray"
            elif controller.finger_held[i]:
                status, color = "HELD", "green"
            elif controller.gripping:
                status = f"CLOSING ({controller.positions[i]:.0f}\u00b0)"
                color = "orange"
            else:
                status, color = "OPEN", "#1f77b4"
            lines[i].set_color(color)
            status_texts[i].set_text(status)

        if controller.gripping:
            held = sum(controller.finger_held)
            state_text.set_text(
                f"GRIPPING ({held}/{NUM_FINGERS} held)  Threshold: {controller.threshold:.2f} A"
            )
            state_text.set_bbox(
                dict(boxstyle="round", facecolor="lightsalmon", alpha=0.9)
            )
        else:
            state_text.set_text(f"OPEN  Threshold: {controller.threshold:.2f} A")
            state_text.set_bbox(
                dict(boxstyle="round", facecolor="lightgreen", alpha=0.9)
            )

        min_x = max(0, current_time - window_size)
        for ax in axes:
            ax.set_xlim(min_x, min_x + window_size)

    ani = FuncAnimation(
        fig, update_plot, interval=20, blit=False, cache_frame_data=False
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        RUNNING = False
        client.close()


if __name__ == "__main__":
    main()
