import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.patches import Rectangle, FancyBboxPatch


N_ROWS = 8
N_COLS = 4
JOINT_PAIRS = {'Hip flex': {'left': 'hip_flexion_l', 'right': 'hip_flexion_r'}, 'Hip add': {'left': 'hip_adduction_l', 'right': 'hip_adduction_r'}, 'Hip rot': {'left': 'hip_rotation_l', 'right': 'hip_rotation_r'}, 'Knee flex': {'left': 'knee_angle_l', 'right': 'knee_angle_r'}, 'Knee rot': {'left': 'knee_angle_l_rotation2', 'right': 'knee_angle_r_rotation2'}, 'Ankle': {'left': 'ankle_angle_l', 'right': 'ankle_angle_r'}, 'Subtalar': {'left': 'subtalar_angle_l', 'right': 'subtalar_angle_r'}, 'MTP': {'left': 'mtp_angle_l', 'right': 'mtp_angle_r'}}
MUSCLE_ORDER = ['Iliopsoas', 'Quadriceps', 'Adductor', 'Adductor magnus', 'Biceps femoris', 'Tibialis anterior', 'Gastrocnemius', 'Extensor hallucis']

def electrode_to_position(electrode_id):
    """
    Paddle numbering:

         1    9   17   25
         2   10   18   26
         3   11   19   27
         4   12   20   28
         5   13   21   29
         6   14   22   30
         7   15   23   31
         8   16   24   32
    """
    electrode_id = int(electrode_id)
    col = (electrode_id - 1) // N_ROWS
    row = (electrode_id - 1) % N_ROWS
    return (row, col)

def plot_paddle_pattern_ax(ax, pattern, electrode_width=0.42, electrode_height=0.72):

    active_sites = pattern["sites"]

    x_positions = {0: -1.15, 1: -0.38, 2: 0.38, 3: 1.15}

    # Paddle centered around the electrode array
    paddle_width = 3.4
    paddle_height = N_ROWS + 1.8
    electrode_center_y = (N_ROWS - 1) / 2 + 0.14

    x0 = -paddle_width / 2
    y0 = electrode_center_y - paddle_height / 2

    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), paddle_width, paddle_height,
            boxstyle="round,pad=0.03,rounding_size=0.15",
            facecolor="#E5E5E5",
            alpha=0.7,
            edgecolor="#A8A8A8",
            linewidth=1.5,
        )
    )

    for electrode_id in range(1, 33):

        row, col = electrode_to_position(electrode_id)

        x = x_positions[col]
        y = N_ROWS - 1 - row + (0.28 if col in [1, 2] else 0)

        active = electrode_id in active_sites

        ax.add_patch(
            Rectangle(
                (x - electrode_width / 2, y - electrode_height / 2),
                electrode_width,
                electrode_height,
                facecolor="#5FBF72" if active else "#6F6F6F",
                edgecolor="#429957" if active else "#505050",
                alpha=0.7,
                linewidth=1,
            )
        )

        ax.text(
            x, y + (0.09 if active else 0),
            str(electrode_id),
            ha="center", va="center",
            fontsize=9,
            fontweight="bold" if active else "normal",
            color="black" if active else "white",
        )

        if active:
            ax.text(
                x, y - 0.18,
                f"{active_sites[electrode_id]:.2f}",
                ha="center", va="center",
                fontsize=7,
            )

    ax.set(
        xlim=(-2.1, 2.1),
        ylim=(y0 - 0.2, y0 + paddle_height + 0.2),
        xticks=[],
        yticks=[],
        aspect="equal",
    )

    ax.set_title("Paddle configuration", fontsize=18)

    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_muscle_activation_ax(ax, muscle_df):
    """
    Plot left/right steady-state muscle activation.
    """
    muscles = [muscle for muscle in MUSCLE_ORDER if muscle in muscle_df['muscle'].unique()]
    left_values = []
    right_values = []
    for muscle in muscles:
        left = muscle_df[(muscle_df['muscle'] == muscle) & (muscle_df['side'] == 'left')]
        right = muscle_df[(muscle_df['muscle'] == muscle) & (muscle_df['side'] == 'right')]
        left_values.append(left['muscle_recruitment'].iloc[0] if len(left) else 0.0)
        right_values.append(right['muscle_recruitment'].iloc[0] if len(right) else 0.0)
    y = np.arange(len(muscles))
    bar_height = 0.36
    bars_left = ax.barh(y - bar_height / 2, left_values, height=bar_height, label='Left', alpha=0.8)
    bars_right = ax.barh(y + bar_height / 2, right_values, height=bar_height, label='Right', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(muscles)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Tetanic muscle activation', fontsize=13)
    ax.set_title('Muscle activation', fontsize=18)
    ax.tick_params("both", labelsize=12)
    ax.grid(axis='x', alpha=0.2)
    for bar, value in zip(bars_left, left_values):
        if value > 0:
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center', fontsize=12)
    for bar, value in zip(bars_right, right_values):
        if value > 0:
            ax.text(value + 0.01, bar.get_y() + bar.get_height() / 2, f'{value:.2f}', va='center', fontsize=12)
            

def plot_joint_kinematics_ax(ax, joint_summary):
    """
    Plot maximum joint excursion.
    """
    if joint_summary.empty:
        ax.text(0.5, 0.5, 'No joint data', ha='center', va='center')
        ax.axis('off')
        return
    joint_labels = []
    left_values = []
    right_values = []
    for label, pair in JOINT_PAIRS.items():
        left_row = joint_summary[joint_summary['joint'] == pair['left']]
        right_row = joint_summary[joint_summary['joint'] == pair['right']]
        left_values.append(left_row['max_excursion_deg'].iloc[0] if len(left_row) else 0.0)
        right_values.append(right_row['max_excursion_deg'].iloc[0] if len(right_row) else 0.0)
        joint_labels.append(label)
    y = np.arange(len(joint_labels))
    bar_height = 0.36
    bars_left = ax.barh(y - bar_height / 2, left_values, height=bar_height, label='Left', alpha=0.8)
    bars_right = ax.barh(y + bar_height / 2, right_values, height=bar_height, label='Right', alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(joint_labels)
    ax.spines.top.set_visible(False)
    ax.spines.right.set_visible(False)
    ax.invert_yaxis()
    ax.set_xlabel('Maximum excursion [deg]', fontsize=13)
    ax.set_title('Joint kinematics', fontsize=18)
    ax.legend(frameon=False, fontsize=13)
    ax.tick_params("both", labelsize=12)
    ax.grid(axis='x', alpha=0.2)
    xmax = max(max(left_values + right_values) * 1.2, 1.0)
    ax.set_xlim(0, xmax)
    for bar, value in zip(bars_left, left_values):
        if value > 0:
            ax.text(value, bar.get_y() + bar.get_height() / 2, f' {value:.1f}°', va='center', fontsize=12)
    for bar, value in zip(bars_right, right_values):
        if value > 0:
            ax.text(value, bar.get_y() + bar.get_height() / 2, f' {value:.1f}°', va='center', fontsize=12)

def plot_pattern_summary(pattern_name, pattern, result):
    fig = plt.figure(figsize=(18, 8), dpi=180)
    gs = fig.add_gridspec(nrows=1, ncols=3, width_ratios=[1, 1, 1], wspace=0.45)
    ax_paddle = fig.add_subplot(gs[0, 0])
    ax_muscles = fig.add_subplot(gs[0, 1])
    ax_joints = fig.add_subplot(gs[0, 2])
    plot_paddle_pattern_ax(ax=ax_paddle, pattern=pattern)
    plot_muscle_activation_ax(ax=ax_muscles, muscle_df=result['muscles'])
    plot_joint_kinematics_ax(ax=ax_joints, joint_summary=result['joint_summary'])
    current_text = ', '.join([f'Active cathode #{site} | I = {current:.1f} mA' for site, current in pattern['sites'].items()])
    fig.suptitle(f'{current_text}', fontsize=20, y=1.02)
    plt.show()


def plot_rootlet_extracellular_potentials(
    fiber_activation_df,
    threshold_mv,
    figsize=(16, 5),
    dpi=150,
):
    threshold_color = "#F4A261"
    active_color = "#5FBF72"
    inactive_color = "#D3D3D3"

    df_plot = fiber_activation_df.copy()

    df_plot["target"] = (
        df_plot["segment"]
        + "_"
        + df_plot["rootlet"].astype(str)
        + "_"
        + df_plot["side"]
    )

    rootlets = (
        df_plot[["segment", "rootlet", "side", "target"]]
        .drop_duplicates()
        .sort_values(["side", "segment", "rootlet"])
        .reset_index(drop=True)
    )

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # ------------------------------------------------------------
    # Fibers
    # ------------------------------------------------------------

    rng = np.random.default_rng(0)

    for i, row in rootlets.iterrows():

        values = df_plot.loc[
            df_plot["target"] == row["target"],
            "abs_potential_mv",
        ].values

        jitter = rng.uniform(-0.28, 0.28, len(values))

        colors = np.where(
            values >= threshold_mv,
            active_color,
            inactive_color,
        )

        ax.scatter(
            i + jitter,
            values,
            c=colors,
            s=10,
            alpha=0.7,
            edgecolors="none",
        )

    # ------------------------------------------------------------
    # Threshold
    # ------------------------------------------------------------

    ax.axhline(
        threshold_mv,
        linestyle="--",
        linewidth=2,
        color=threshold_color,
        label=f"Activation threshold ({threshold_mv:.1f} mV)",
    )

    # ------------------------------------------------------------
    # Rootlet numbers
    # ------------------------------------------------------------

    ax.set_xticks(np.arange(len(rootlets)))
    ax.set_xticklabels(
        rootlets["rootlet"].astype(str),
        fontsize=9,
    )

    # ------------------------------------------------------------
    # Segment labels
    # ------------------------------------------------------------

    transform = ax.get_xaxis_transform()

    for (side, segment), group in rootlets.groupby(
        ["side", "segment"],
        sort=False,
    ):

        start = group.index.min()
        end = group.index.max()
        center = (start + end) / 2

        # Segment name once, centered below its rootlets
        ax.text(
            center,
            -0.11,
            segment,
            ha="center",
            va="top",
            fontsize=11,
            fontweight="bold",
            transform=transform,
            clip_on=False,
        )

        # Separator between segments
        if end < len(rootlets) - 1:
            ax.axvline(
                end + 0.5,
                color="0.85",
                linewidth=1,
                zorder=0,
            )

    # ------------------------------------------------------------
    # Side labels
    # ------------------------------------------------------------

    for side, group in rootlets.groupby("side", sort=False):

        start = group.index.min()
        end = group.index.max()
        center = (start + end) / 2

        ax.text(
            center,
            -0.21,
            side.capitalize(),
            ha="center",
            va="top",
            fontsize=12,
            transform=transform,
            clip_on=False,
        )

    # Stronger separator between left and right
    left_end = rootlets.loc[rootlets["side"] == "left"].index.max()

    ax.axvline(
        left_end + 0.5,
        color="0.65",
        linewidth=1.5,
        zorder=0,
    )

    ax.set_xlabel("Dorsal rootlets", fontsize=15, labelpad=45)
    ax.set_ylabel(
        "|Extracellular potential| [mV]",
        fontsize=15,
        labelpad=10,
    )

    ax.set_title(
        "Sensory-fiber extracellular potentials across dorsal rootlets",
        fontsize=18,
        y=1.05
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="y", labelsize=11)

    ax.legend(
        frameon=False,
        fontsize=13,
    )

    plt.subplots_adjust(bottom=0.27)
    plt.show()