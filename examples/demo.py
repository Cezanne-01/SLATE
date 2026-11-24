"""SLATE on one recording with the tablet-like degradation of the article applied.

The recording is one trial of the public ETDD70 dataset (a child reading a seven-line text
aloud, 250 Hz, see THIRD_PARTY_NOTICES.md). The functions below are used by examples/demo.ipynb.
Run as a script, this module prints the accuracy of the nearest text line and of SLATE and
writes the figures and the animation next to it.

    python -m examples.demo
"""

import base64
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle

from analysis.public import degradation, evaluation
from analysis.public.run import SEED
from slate import assign_lines, correct_fixations, detect_line_transitions

HERE = Path(__file__).resolve().parent
# line colors, text-line bars and outlines as in the figures of the article
LINE_COLORS = LinearSegmentedColormap.from_list(
    "lines",
    ["#b22222", "#d2691e", "#c49a22", "#8aa23a", "#3f9a52", "#2a8f86", "#2a6f9e", "#1f3f8f"],
)
BAR, GRAY, UNKNOWN = "#d4d4d4", "#8f8f8f", "#555555"


def load_trial(path=HERE / "etdd70_trial.json"):
    """The example trial: samples (t, x, y), fixations with their line labels, line and word
    positions."""
    trial = json.loads(Path(path).read_text())
    trial["samples"] = {k: np.array(v, float) for k, v in trial["samples"].items()}
    for key in ("fixations", "line_Y", "line_X", "word_XY"):
        trial[key] = np.array(trial[key], float)
    return trial


def text_lines(trial):
    """The text lines of the trial as `correct_fixations` expects them."""
    words = trial["word_XY"]
    lines = []
    for (left, right), y in zip(trial["line_X"], trial["line_Y"]):
        centers = words[words[:, 1] == y, 0]
        # the last word ends at the right edge of the line, so its width is twice the distance
        # from its center to that edge
        lines.append(
            dict(left=left, right=right, y=y, n_word=len(centers), last_w=2 * (right - centers.max()))
        )
    return lines


def run(trial, signal):
    """Assign the fixations of the trial to lines under one signal condition ("original" or
    "tablet-like"), by the nearest text line and by SLATE.

    Returns a dict with the samples of the condition (t, x, y), the fixations placed at the mean
    of those samples (`observed`), the labeled line of each fixation (`truth`), the line
    transitions found by SLATE, the table returned by `slate.correct_fixations` (`table`), the
    line of each fixation by both methods (`nearest`, `slate`), the fixation positions after SLATE
    (`corrected`) and the accuracy of both methods."""
    line_Y, widths, pitch, text_width, text_cx = evaluation.geometry(trial)
    fixations = trial["fixations"]
    truth = fixations[:, 4].astype(int)
    samples = trial["samples"]
    t, x, y = degradation.make_signal(
        samples["t"], samples["x"], samples["y"], signal, pitch, text_width, text_cx,
        float(np.mean(line_Y)), SEED, evaluation.trial_seed(trial["trial_id"]),
        ev=evaluation.reference_transitions(fixations),
    )
    observed = evaluation.fix_from_samples(fixations, t, x, y)

    nearest = np.argmin(np.abs(line_Y[None, :] - observed[:, 3][:, None]), axis=1) + 1
    transitions = detect_line_transitions(t, x, line_widths=widths, first_at_start=True)
    table = correct_fixations(observed, t, x, text_lines(trial), transitions=transitions)
    slate = table.line.to_numpy()
    corrected = table[["start", "end", "x_corrected", "y_corrected"]].to_numpy()

    def accuracy(assigned):
        return float(np.mean((assigned == truth) & (truth > 0)))

    return dict(
        t=t, x=x, y=y, observed=observed, corrected=corrected, truth=truth, table=table,
        transitions=transitions, nearest=nearest, slate=slate,
        accuracy=dict(nearest=accuracy(nearest), slate=accuracy(slate)),
    )


def color_of(lines, n_lines):
    """Color of each line number (1-based), gray for 0 (no line)."""
    lines = np.asarray(lines)
    colors = LINE_COLORS((lines - 1) / max(n_lines - 1, 1))
    colors[lines < 1] = (0.6, 0.6, 0.6, 1.0)
    return colors


def draw_page(ax, trial, positions, colors, title=None, numbers=False):
    """Fixations on the page: gray bars are the text lines, dots the fixations (size: duration)."""
    line_Y, line_X = trial["line_Y"], trial["line_X"]
    pitch = float(np.median(np.diff(line_Y)))
    for (left, right), line_y in zip(line_X, line_Y):
        ax.add_patch(
            Rectangle((left, line_y - 0.15 * pitch), right - left, 0.3 * pitch, fc=BAR, ec="none")
        )
    duration = positions[:, 1] - positions[:, 0]
    ax.plot(positions[:, 2], positions[:, 3], color=GRAY, lw=0.35, alpha=0.8, zorder=2)
    ax.scatter(
        positions[:, 2], positions[:, 3], s=8 + 30 * np.clip(duration, 0, 1.2), c=colors,
        edgecolors="white", linewidths=0.3, zorder=3,
    )
    width = line_X[:, 1].max() - line_X[:, 0].min()
    ax.set_xlim(line_X[:, 0].min() - 0.16 * width, line_X[:, 1].max() + 0.16 * width)
    ax.set_ylim(line_Y[-1] + 2.6 * pitch, line_Y[0] - 2.6 * pitch)
    if numbers:
        for k, line_y in enumerate(line_Y, 1):
            ax.text(
                line_X[:, 0].min() - 0.035 * width, line_y, str(k), ha="right", va="center",
                color=color_of([k], len(line_Y))[0], fontsize=9, fontweight="bold",
            )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRAY)
        spine.set_linewidth(0.7)
    if title:
        ax.set_title(title, fontsize=10, color="#444444", pad=4)


def figure_degradation(trial, original, tablet):
    """The recording as published and with the tablet-like degradation applied. Color: the
    labeled line of each fixation."""
    n = len(trial["line_Y"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    draw_page(axes[0], trial, original["observed"], color_of(original["truth"], n),
              "Recording as published (250 Hz)", numbers=True)
    draw_page(axes[1], trial, tablet["observed"], color_of(tablet["truth"], n),
              "Tablet-like signal (29.4 Hz)")
    return fig


def figure_transitions(trial, result):
    """Horizontal coordinate over time with the line transitions found by SLATE, and the line of
    each fixation as labeled in the dataset, by SLATE and by the nearest text line."""
    n = len(trial["line_Y"])
    fig, (trace, ribbons) = plt.subplots(
        2, 1, figsize=(7.4, 3.9), sharex=True, constrained_layout=True,
        gridspec_kw=dict(height_ratios=[3, 1.15]),
    )
    t, x = result["t"], result["x"]
    line_of_sample = assign_lines(t, result["transitions"], n_lines=n)
    for k in range(1, n + 1):
        own = np.where(line_of_sample == k, x, np.nan)
        trace.plot(t, own, color=color_of([k], n)[0], lw=0.9)
        inside = t[line_of_sample == k]
        if len(inside):
            trace.text(inside.mean(), np.nanmax(x) * 1.03, str(k), color=color_of([k], n)[0],
                       ha="center", va="bottom", fontsize=9, fontweight="bold")
    for time in result["transitions"][1:]:
        trace.axvline(time, color=GRAY, lw=0.6, ls=(0, (3, 2)))
    trace.set_ylabel("horizontal x (px)")
    trace.set_ylim(top=np.nanmax(x) * 1.14)

    start, end = result["observed"][:, 0], result["observed"][:, 1]
    rows = (
        ("labeled", result["truth"]),
        ("SLATE", result["slate"]),
        ("nearest line", result["nearest"]),
    )
    for row, (_, assigned) in enumerate(rows):
        ribbons.barh(
            np.full(len(start), -row), end - start, left=start, height=0.7,
            color=color_of(assigned, n), linewidth=0,
        )
    ribbons.set_yticks([0, -1, -2])
    ribbons.set_yticklabels([name for name, _ in rows])
    ribbons.set_xlabel("time (s)")
    ribbons.tick_params(axis="y", length=0)
    for ax in (trace, ribbons):
        ax.spines[["top", "right"]].set_visible(False)
    ribbons.spines["left"].set_visible(False)
    return fig


def figure_before_after(trial, result):
    """The tablet-like recording before and after SLATE: the recorded fixation positions, and
    the same fixations on their assigned line with the recalibrated horizontal position."""
    n = len(trial["line_Y"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    draw_page(axes[0], trial, result["observed"], [UNKNOWN] * len(result["observed"]),
              "Recorded (tablet-like signal)")
    draw_page(axes[1], trial, result["corrected"], color_of(result["slate"], n),
              "After SLATE (color: assigned line)", numbers=True)
    return fig


def animate_before_after(trial, result, path, speed=1.0, fps=10, trail=12):
    """Replay of the tablet-like recording before and after SLATE, written as a GIF. `speed` is
    the playback rate relative to real time. The last `trail` fixations stay visible."""
    n = len(trial["line_Y"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0), constrained_layout=True)
    panels = []
    for ax, positions, colors, title in (
        (axes[0], result["observed"], np.array([[0.33, 0.33, 0.33, 1.0]] * len(result["slate"])),
         "Recorded (tablet-like signal)"),
        (axes[1], result["corrected"], color_of(result["slate"], n),
         "After SLATE (color: assigned line)"),
    ):
        draw_page(ax, trial, positions[:0], [], title, numbers=ax is axes[1])
        dots = ax.scatter([], [], s=55, edgecolors="white", linewidths=0.4, zorder=3)
        (path_line,) = ax.plot([], [], color=GRAY, lw=0.6, zorder=2)
        panels.append((positions, colors, dots, path_line))
    clock = fig.text(0.5, 0.02, "", ha="center", va="bottom", fontsize=9, color="#444444")
    start = result["observed"][:, 0]
    times = np.arange(0.0, result["t"][-1] + 1.0, speed / fps)

    def frame(now):
        shown = np.flatnonzero(start <= now)[-trail:]
        for positions, colors, dots, path_line in panels:
            dots.set_offsets(positions[shown, 2:4])
            faded = colors[shown].copy()
            faded[:, 3] = np.linspace(0.25, 1.0, len(shown)) if len(shown) else []
            dots.set_facecolors(faded)
            path_line.set_data(positions[shown, 2], positions[shown, 3])
        clock.set_text(f"{now:4.1f} s")
        return []

    FuncAnimation(fig, frame, frames=times).save(path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return path


def embedded(path):
    """The GIF at `path` as an HTML image with the file contents inside, for showing it in a
    notebook."""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/gif;base64,{data}" alt="Replay of the reading">'


def main():
    trial = load_trial()
    original, tablet = run(trial, "original"), run(trial, "tablet-like")
    print(f"{'signal':14s} {'nearest line':>13s} {'SLATE':>8s}")
    for name, r in (("original", original), ("tablet-like", tablet)):
        nearest, slate = 100 * r["accuracy"]["nearest"], 100 * r["accuracy"]["slate"]
        print(f"{name:14s} {nearest:12.1f}% {slate:7.1f}%")
    for name, fig in (
        ("degradation", figure_degradation(trial, original, tablet)),
        ("transitions", figure_transitions(trial, tablet)),
        ("before_after", figure_before_after(trial, tablet)),
    ):
        fig.savefig(HERE / f"{name}.png", dpi=200, facecolor="white")
        plt.close(fig)
    animate_before_after(trial, tablet, HERE / "before_after.gif", speed=2.0)
    print("wrote degradation.png, transitions.png, before_after.png and before_after.gif to", HERE)


if __name__ == "__main__":
    plt.switch_backend("Agg")
    main()
