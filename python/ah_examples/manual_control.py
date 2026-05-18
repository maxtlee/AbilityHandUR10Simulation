import time
import threading
from collections import deque

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import gridspec

# Suppress all default matplotlib key bindings (s=save, q=quit, etc.)
# so they don't conflict with our joint controls.
for key in list(mpl.rcParams):
    if key.startswith("keymap."):
        mpl.rcParams[key] = []

from ah_wrapper import AHSerialClient

JOINT_NAMES = ["Index", "Middle", "Ring", "Pinky", "Thb Flex", "Thb Rot"]
FINGER_NAMES = ["Index", "Middle", "Ring", "Pinky", "Thumb"]
NUM_JOINTS = 6
NUM_FINGERS = 5
OPEN_POS = [30, 30, 30, 30, 30, -50]
MOVE_RATE = 50  # degrees per second

# Position limits per joint: (min, max)
POS_LIMITS = [
    (0, 100), (0, 100), (0, 100), (0, 100),  # index, middle, ring, pinky
    (0, 100),   # thumb flexor
    (-100, 0),  # thumb rotator
]

# Key -> (joint_index, direction)
# Bottom-row keys close, top-row keys open
# Left to right: qa=pinky, ws=ring, ed=middle, rf=index, uj=thb_flex, ik=thb_rot
KEY_MAP = {
    'a': (3, +1), 'q': (3, -1),   # pinky
    's': (2, +1), 'w': (2, -1),   # ring
    'd': (1, +1), 'e': (1, -1),   # middle
    'f': (0, +1), 'r': (0, -1),   # index
    'j': (4, +1), 'u': (4, -1),   # thumb flexor
    'k': (5, -1), 'i': (5, +1),   # thumb rotator
}

RUNNING = True


class ManualController:
    """Holds target positions for all 6 joints and applies key-driven
    increments each control tick. Alternates reply modes 0 and 1 so that
    position, current, velocity, and FSR feedback are all available."""

    def __init__(self, client):
        self.client = client
        self.positions = list(OPEN_POS)
        self.keys_pressed = set()
        self._alternate = True

    def open_hand(self):
        self.positions = list(OPEN_POS)

    def update(self, dt):
        for key in list(self.keys_pressed):
            if key in KEY_MAP:
                idx, direction = KEY_MAP[key]
                lo, hi = POS_LIMITS[idx]
                self.positions[idx] = max(
                    lo, min(hi, self.positions[idx] + direction * MOVE_RATE * dt)
                )

        # Alternate: mode 0 = pos+cur+fsr, mode 1 = pos+vel+fsr
        reply_mode = 0 if self._alternate else 1
        self._alternate = not self._alternate
        self.client.set_position(
            positions=list(self.positions), reply_mode=reply_mode
        )


def control_thread(controller):
    dt = 1.0 / controller.client.rate_hz
    while RUNNING:
        controller.update(dt)
        controller.client.send_command()
        time.sleep(dt)


def main():
    global RUNNING

    client = AHSerialClient(write_thread=False)
    controller = ManualController(client)

    ctrl_thread = threading.Thread(target=control_thread, args=(controller,))
    ctrl_thread.start()
    time.sleep(0.5)

    # --- Data buffers ---
    window_size = 10
    max_samples = window_size * 50
    x_data = deque(maxlen=max_samples)
    pos_data = [deque(maxlen=max_samples) for _ in range(NUM_JOINTS)]
    vel_data = [deque(maxlen=max_samples) for _ in range(NUM_JOINTS)]
    cur_data = [deque(maxlen=max_samples) for _ in range(NUM_JOINTS)]
    fsr_data = [deque(maxlen=max_samples) for _ in range(NUM_FINGERS)]
    start_time = time.time()

    # --- Figure layout ---
    fig = plt.figure(figsize=(14, 9))
    fig.canvas.manager.set_window_title("PSYONIC Manual Control")

    gs_main = gridspec.GridSpec(
        3, 2, figure=fig, width_ratios=[1.2, 1], wspace=0.35, hspace=0.45,
    )

    # Left column: 3 motor plots (position, velocity, current)
    motor_axes = [fig.add_subplot(gs_main[i, 0]) for i in range(3)]
    for ax in motor_axes[:-1]:
        ax.tick_params(labelbottom=False)

    # Right column: 5 touch sensor plots nested inside the full right column
    gs_touch = gridspec.GridSpecFromSubplotSpec(
        5, 1, subplot_spec=gs_main[:, 1], hspace=0.5,
    )
    touch_axes = [fig.add_subplot(gs_touch[i]) for i in range(5)]
    for ax in touch_axes[:-1]:
        ax.tick_params(labelbottom=False)

    # --- Motor plot setup ---
    motor_titles = ["Position", "Velocity", "Current"]
    motor_ylabels = ["\u00b0", "\u00b0/s", "A"]
    motor_yranges = [(0, 110), (-500, 500), (-1, 1)]
    motor_lines = []  # motor_lines[plot_idx][joint_idx]

    for plot_idx, ax in enumerate(motor_axes):
        jlines = []
        for j in range(NUM_JOINTS):
            (ln,) = ax.plot([], [], linewidth=1)
            jlines.append(ln)
        motor_lines.append(jlines)
        ax.set_ylim(*motor_yranges[plot_idx])
        ax.set_ylabel(motor_ylabels[plot_idx], fontsize=9)
        ax.set_title(
            motor_titles[plot_idx], fontsize=10, loc="left", fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
    motor_axes[-1].set_xlabel("Time (s)")
    fig.legend(
        motor_lines[0], JOINT_NAMES, loc="upper left",
        fontsize=7, ncol=6, bbox_to_anchor=(0.02, 0.99),
    )

    # --- Touch plot setup ---
    touch_lines = []
    for i, ax in enumerate(touch_axes):
        (ln,) = ax.plot([], [], linewidth=1.5, color="#1f77b4")
        touch_lines.append(ln)
        ax.set_ylim(0, 8)
        ax.set_ylabel("N", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            FINGER_NAMES[i], fontsize=9, loc="left", fontweight="bold",
        )
    touch_axes[0].text(
        0.5, 1.35, "Touch Sensors", transform=touch_axes[0].transAxes,
        ha="center", fontsize=10, fontweight="bold",
    )
    touch_axes[-1].set_xlabel("Time (s)")

    # --- Annotations ---
    fig.text(
        0.5, 0.005,
        "[Q/A] Pinky  [W/S] Ring  [E/D] Middle  [R/F] Index  "
        "[U/J] Thb Flex  [I/K] Thb Rot  [Space] Open  [Esc] Quit",
        ha="center", fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.9),
    )
    # Axes-level text so it can be returned as a blit artist
    pos_text = motor_axes[0].text(
        0.5, 1.15, "", transform=motor_axes[0].transAxes,
        fontsize=8, ha="center", family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.9),
    )

    plt.subplots_adjust(bottom=0.06, top=0.94, left=0.06, right=0.98)

    # --- Keyboard ---
    def on_key_press(event):
        global RUNNING
        if event.key == ' ':
            controller.open_hand()
        elif event.key == 'escape':
            RUNNING = False
            plt.close(fig)
        else:
            controller.keys_pressed.add(event.key)

    def on_key_release(event):
        controller.keys_pressed.discard(event.key)

    fig.canvas.mpl_connect("key_press_event", on_key_press)
    fig.canvas.mpl_connect("key_release_event", on_key_release)

    # Collect all animated artists for blit
    all_artists = []
    for jlines in motor_lines:
        all_artists.extend(jlines)
    all_artists.extend(touch_lines)
    all_artists.append(pos_text)

    # --- Animation ---
    def update_plot(frame):
        current_time = time.time() - start_time
        x_data.append(current_time)

        hand = client.hand
        pos = hand.get_position()
        vel = hand.get_velocity()
        cur = hand.get_current()
        fsr = hand.get_fsr()

        for j in range(NUM_JOINTS):
            p = pos[j] if pos else 0.0
            # Negate thumb rotator for display (matches RealTimePlotMotors)
            pos_data[j].append(-p if j == 5 else p)
            vel_data[j].append(vel[j] if vel else 0.0)
            cur_data[j].append(cur[j] if cur else 0.0)

        for i in range(NUM_FINGERS):
            fsr_data[i].append(sum(fsr[i * 6 : i * 6 + 6]) if fsr else 0.0)

        sources = [pos_data, vel_data, cur_data]
        for plot_idx in range(3):
            for j in range(NUM_JOINTS):
                motor_lines[plot_idx][j].set_data(x_data, sources[plot_idx][j])

        for i in range(NUM_FINGERS):
            touch_lines[i].set_data(x_data, fsr_data[i])

        min_x = max(0, current_time - window_size)
        for ax in motor_axes + touch_axes:
            ax.set_xlim(min_x, min_x + window_size)

        t = controller.positions
        pos_text.set_text(
            f"Targets:  Idx={t[0]:5.1f}\u00b0  Mid={t[1]:5.1f}\u00b0  "
            f"Rng={t[2]:5.1f}\u00b0  Pnk={t[3]:5.1f}\u00b0  "
            f"ThF={t[4]:5.1f}\u00b0  ThR={t[5]:6.1f}\u00b0"
        )

        return all_artists

    ani = FuncAnimation(
        fig, update_plot, interval=10, blit=True, cache_frame_data=False,
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
