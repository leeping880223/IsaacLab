# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Plot an OpenArm grip trial CSV without launching Isaac Sim.

Example::

    python scripts/environments/grip_output/openarm_grip_plot.py \
        scripts/environments/grip_output/openarm_official_cylinder_7mm_100g_joint1_only_grip.csv

Run without a CSV argument to pick a file from a dialog. The PNG is saved beside
the CSV. Use ``--env 1`` for a different environment.
"""

import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_rows(csv_path: Path, env_id: int) -> tuple[list[dict[str, str]], set[str]]:
    """Read one environment's time-series rows from a grip CSV."""
    with csv_path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or ())
        required = {"env", "phase", "time_s", "left_normal_N", "right_normal_N"}
        missing = required - columns
        if missing:
            raise ValueError(f"CSV is missing columns: {', '.join(sorted(missing))}")
        rows = [row for row in reader if int(row["env"]) == env_id]
    if not rows:
        raise ValueError(f"CSV has no rows for env {env_id}")
    rows.sort(key=lambda row: float(row["time_s"]))
    return rows, columns


def _values(rows: list[dict[str, str]], column: str, scale: float = 1.0) -> list[float]:
    """Return a numeric CSV column, optionally converted to another unit."""
    return [float(row[column]) * scale for row in rows]


def _phase_markers(ax: plt.Axes, rows: list[dict[str, str]], times: list[float], *, label: bool) -> None:
    """Label each contiguous open, close, lift, or hold interval."""
    starts = [0] + [index for index in range(1, len(rows)) if rows[index]["phase"] != rows[index - 1]["phase"]]
    for phase_index, start in enumerate(starts):
        end = starts[phase_index + 1] if phase_index + 1 < len(starts) else len(rows) - 1
        x0, x1 = times[start], times[end]
        if phase_index % 2:
            ax.axvspan(x0, x1, color="0.92", alpha=0.5, zorder=0)
        if phase_index:
            ax.axvline(x0, color="0.55", linestyle=":", linewidth=1)
        if label:
            ax.text(
                (x0 + x1) / 2,
                0.98,
                rows[start]["phase"],
                ha="center",
                va="top",
                transform=ax.get_xaxis_transform(),
            )


def plot_csv(csv_path: Path, env_id: int, output: Path) -> None:
    """Save force, motion, joint, drive-estimate, and gravity plots to a PNG."""
    rows, columns = _read_rows(csv_path, env_id)
    times = _values(rows, "time_s")
    height_column = "cylinder_z_m" if "cylinder_z_m" in columns else "bar_z_m"
    panels = ["contact"]
    if {height_column, "hand_z_m"} <= columns:
        panels.append("height")
    if "finger_joint1_m" in columns:
        panels.append("joints")
    if "finger_joint1_drive_est_N" in columns:
        panels.append("drive")
    if "gravity_m_s2" in columns:
        panels.append("gravity")

    fig, axes = plt.subplots(len(panels), 1, sharex=True, figsize=(13, 2.6 * len(panels) + 1.5), squeeze=False)
    axes = axes[:, 0]
    for ax, panel in zip(axes, panels):
        ax.grid(True, alpha=0.3)
        if panel == "contact":
            ax.plot(times, _values(rows, "left_normal_N"), label="Left finger", color="tab:blue")
            ax.plot(times, _values(rows, "right_normal_N"), label="Right finger", color="tab:orange")
            ax.set_ylabel("Contact force [N]")
        elif panel == "height":
            ax.plot(times, _values(rows, height_column), label="Cylinder", color="tab:green")
            ax.plot(times, _values(rows, "hand_z_m"), label="Hand", color="tab:purple")
            ax.set_ylabel("Height [m]")
        elif panel == "joints":
            for index, color in ((1, "tab:blue"), (2, "tab:orange")):
                actual = f"finger_joint{index}_m"
                target = f"finger_joint{index}_target_m"
                if actual in columns:
                    ax.plot(times, _values(rows, actual, 1000), label=f"Joint {index}", color=color)
                if target in columns:
                    ax.plot(
                        times, _values(rows, target, 1000), label=f"Joint {index} target", color=color, linestyle="--"
                    )
            ax.set_ylabel("Finger position [mm]")
        elif panel == "drive":
            for index, color in ((1, "tab:blue"), (2, "tab:orange")):
                column = f"finger_joint{index}_drive_est_N"
                if column in columns:
                    ax.plot(times, _values(rows, column), label=f"Joint {index} estimate", color=color)
            ax.set_ylabel("Drive estimate [N]")
        elif panel == "gravity":
            ax.plot(times, _values(rows, "gravity_m_s2"), label="Gravity", color="tab:gray")
            ax.set_ylabel("Gravity [m/s²]")
            if "object_gravity_N" in columns:
                mass_kg = next(
                    (
                        float(row["object_gravity_N"]) / float(row["gravity_m_s2"])
                        for row in rows
                        if float(row["gravity_m_s2"]) > 0
                    ),
                    None,
                )
                if mass_kg and mass_kg > 0:
                    weight_ax = ax.secondary_yaxis(
                        "right", functions=(lambda gravity: gravity * mass_kg, lambda weight: weight / mass_kg)
                    )
                    weight_ax.set_ylabel("Object weight [N]", color="tab:red")
                    weight_ax.tick_params(axis="y", colors="tab:red")
        _phase_markers(ax, rows, times, label=panel == "contact")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Simulation time [s]")
    fig.suptitle(
        f"{csv_path.stem} — env {env_id}\nDrive force is an actuator estimate, not measured finger contact force",
        fontsize=11,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _choose_csv() -> Path:
    """Choose a CSV in a file dialog, or from a numbered terminal list."""
    directory = Path(__file__).resolve().parent
    if (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")) and shutil.which("zenity"):
        result = subprocess.run(
            [
                "zenity",
                "--file-selection",
                "--title=Choose grip CSV",
                f"--filename={directory}/",
                "--file-filter=*.csv",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return Path(result.stdout.strip())
        if result.returncode == 1 and not result.stderr.strip():
            raise ValueError("CSV selection cancelled")

    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        try:
            selected = filedialog.askopenfilename(
                parent=root, title="Choose grip CSV", initialdir=directory, filetypes=[("CSV files", "*.csv")]
            )
        finally:
            root.destroy()
        if selected:
            return Path(selected)
        raise ValueError("CSV selection cancelled")
    except ImportError:
        pass
    except tk.TclError:
        pass

    files = sorted(directory.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not files:
        raise ValueError(f"No CSV files found in {directory}")
    print("Choose a CSV file (newest first):")
    for index, path in enumerate(files, 1):
        print(f"{index:2d}. {path.name}")
    try:
        choice = int(input("File number: "))
    except (EOFError, ValueError) as exc:
        raise ValueError("Enter a file number, or pass the CSV path as an argument") from exc
    if not 1 <= choice <= len(files):
        raise ValueError(f"File number must be between 1 and {len(files)}")
    return files[choice - 1]


def main() -> None:
    """Parse command-line arguments and write the grip plot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, help="Grip trial CSV; omit to choose a file interactively.")
    parser.add_argument("--env", type=int, default=0, help="Environment index to plot (default: 0).")
    parser.add_argument("--output", type=Path, help="PNG path (default: beside CSV, with _envN_plot suffix).")
    args = parser.parse_args()
    if args.env < 0:
        parser.error("--env must be nonnegative")
    try:
        csv_path = args.csv or _choose_csv()
        if not csv_path.is_file():
            parser.error(f"CSV does not exist: {csv_path}")
        output = args.output or csv_path.with_name(f"{csv_path.stem}_env{args.env}_plot.png")
        plot_csv(csv_path, args.env, output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"Saved plot: {output}")


if __name__ == "__main__":
    main()
