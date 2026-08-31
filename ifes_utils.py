from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d
from tqdm.auto import tqdm
import mujoco

FILE_PATTERN = re.compile(r"site_(\d+)_root_([LS]\d+)_(\d+)_dorsal_(left|right)\.txt")
FEM_REFERENCE_CURRENT_MA = 1000.0  # COMSOL reference current: 1 A

SEGMENT_NAMES = ["L1", "L2", "L3", "L4", "L5", "S1", "S2"]
SEGMENT_LENGTHS_MM = np.array([15.84, 12.32, 10.56, 9.68, 7.48, 6.60, 7.04])
N_ROOTLETS_PER_SEGMENT = np.array([7, 8, 9, 8, 8, 5, 4])

RADIUS_WM_MM = 8.0
RADIUS_CSF_MM = 10.0
MODEL_LENGTH_MM = float(SEGMENT_LENGTHS_MM.sum())

ROOTLET_X_MM = RADIUS_WM_MM * np.cos(np.pi / 3)
ROOTLET_Y_MM = RADIUS_WM_MM * np.sin(np.pi / 3)

# ============================================================
# Geometry
# ============================================================

def electrode_geometry():
    width = 10.0
    height_elec = 53.12
    dist_long = 3.2
    width_elec = 3.44

    distance = RADIUS_CSF_MM + 0.5
    space_be = 1.2
    theta = space_be / distance

    x_values = [
        width / 2 - distance * np.sin(1.5 * theta) - 0.5,
        distance * np.sin(theta / 2) + 0.5,
    ]
    y_values = [np.sqrt(distance**2 - x**2) for x in x_values]

    step = width_elec + dist_long
    z_14 = width_elec / 2 + np.arange(8) * step
    z_23 = width_elec / 2 + dist_long + np.arange(8) * step
    offset = (MODEL_LENGTH_MM - height_elec) / 2

    points = []
    for col_idx, (x, y) in enumerate(zip(x_values * 2, y_values * 2), start=1):
        z_values = z_14 + offset if col_idx % 2 else z_23 + offset
        sign = -1 if col_idx < 3 else 1
        points.extend((sign * x, y, z) for z in z_values)

    points = np.asarray(points)

    ordered = np.zeros_like(points)
    ordered[:8] = points[23:15:-1]
    ordered[8:16] = points[31:23:-1]
    ordered[16:24] = points[15:7:-1]
    ordered[24:] = points[7::-1]

    return ordered


def rootlet_centers():
    centers = {}
    z_start = MODEL_LENGTH_MM

    for segment, length, n_rootlets in zip(
        SEGMENT_NAMES, SEGMENT_LENGTHS_MM, N_ROOTLETS_PER_SEGMENT
    ):
        z_center = z_start - length / 2
        half_length = length / 2.5
        centers[segment] = np.linspace(
            z_center - half_length,
            z_center + half_length,
            int(n_rootlets),
        )[::-1]
        z_start -= length

    return centers

def plot_electrical_model(rootlets=False, labels=True, figsize=(6, 9)):
    electrodes = electrode_geometry()

    fig = plt.figure(figsize=figsize, dpi=150)
    ax = fig.add_subplot(111, projection="3d")

    theta = np.linspace(0, 2*np.pi, 60)
    z = np.linspace(0, MODEL_LENGTH_MM, 60)
    T, Z = np.meshgrid(theta, z)


    for radius, alpha in [(RADIUS_WM_MM, 0.1)]:
        ax.plot_surface(
            radius*np.cos(T),
            radius*np.sin(T),
            Z,
            alpha=alpha,
            linewidth=0,
            shade=False,
            label='Pia mater'
        )

    # Electrodes
    electrode_width = 1.0
    electrode_height = 2.7

    for x, y, z_ in electrodes:
        verts = [[
            (x - electrode_width/2, y, z_ - electrode_height/2),
            (x + electrode_width/2, y, z_ - electrode_height/2),
            (x + electrode_width/2, y, z_ + electrode_height/2),
            (x - electrode_width/2, y, z_ + electrode_height/2),
        ]]

        ax.add_collection3d(
            Poly3DCollection(
                verts,
                facecolor="#6F6F6F",
                edgecolor="#505050",
                linewidth=1,
            )
        )

    # Rootlets
    if rootlets:
        centers = rootlet_centers()
        colors = plt.cm.tab10(np.linspace(0, 1, len(SEGMENT_NAMES)))

        for segment, color in zip(SEGMENT_NAMES, colors):
            z_seg = centers[segment]

            ax.scatter(
                np.full(len(z_seg), -ROOTLET_X_MM),
                np.full(len(z_seg), ROOTLET_Y_MM),
                z_seg,
                s=25,
                color=color,
                label=segment,
            )

            ax.scatter(
                np.full(len(z_seg), ROOTLET_X_MM),
                np.full(len(z_seg), ROOTLET_Y_MM),
                z_seg,
                s=25,
                color=color,
            )

    # View
    ax.view_init(elev=30, azim=-85)
    ax.set_box_aspect(
        (2*RADIUS_CSF_MM, 2*RADIUS_CSF_MM, MODEL_LENGTH_MM)
    )

    ax.set(
        xticks=[],
        yticks=[],
        zticks=[],
        xlabel="",
        ylabel="",
        zlabel="",
    )

    ax.grid(False)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()

    # Numbers projected in 2D -> always visible
    if labels:
        for electrode_id, (x, y, z_) in enumerate(electrodes, 1):
            x2, y2, _ = proj3d.proj_transform(
                x, y, z_, ax.get_proj()
            )

            ax.annotate(
                str(electrode_id),
                (x2, y2),
                ha="center",
                va="center",
                fontsize=7,
                color="k",
                zorder=1000,
            )

    legend_elements = [
        Patch(
            facecolor="lightblue",
            edgecolor="none",
            alpha=0.3,
            label="Pia mater",
        )
    ]

    if rootlets:
        legend_elements += [
            Line2D(
                [0], [0],
                marker="o",
                linestyle="",
                markerfacecolor=color,
                markeredgecolor="none",
                markersize=6,
                label=segment,
            )
            for segment, color in zip(SEGMENT_NAMES, colors)
        ]

    ax.legend(
        handles=legend_elements,
        bbox_to_anchor=(1.02, 0.5),
        loc="center left",
        frameon=False,
        fontsize=12,
    )

    plt.tight_layout()
    plt.show()

# ============================================================
# Lead Field Matrix
# ============================================================

def parse_lfm_filename(filename):
    m = FILE_PATTERN.fullmatch(str(filename))
    if m is None:
        return None

    site = int(m.group(1))
    segment = m.group(2)
    rootlet = int(m.group(3))
    side = m.group(4)

    return {
        "site": site,
        "segment": segment,
        "rootlet": rootlet,
        "side": side,
        "target_name": f"{segment}_{rootlet}_dorsal_{side}",
    }


def read_lead_field_file(filepath, value_column=-1):
    data = np.loadtxt(filepath, comments="%")

    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"Unexpected format in {Path(filepath).name}")

    coordinates = data[:, :3].astype(float)      # x, y, z [mm]
    potential_v = data[:, value_column].astype(float)

    valid = np.isfinite(potential_v) & np.all(np.isfinite(coordinates), axis=1)

    return coordinates[valid], potential_v[valid]


def build_lfm(
    raw_folder,
    value_column=-1,
    reference_current_ma=FEM_REFERENCE_CURRENT_MA,
):
    raw_folder = Path(raw_folder)

    records = []

    for fp in raw_folder.rglob("site_*_root_*.txt"):
        info = parse_lfm_filename(fp.name)

        if info is not None:
            records.append({**info, "filepath": fp})

    if not records:
        raise RuntimeError(f"No raw lead-field files found in {raw_folder}")

    site_ids = sorted({r["site"] for r in records})

    # Anatomical ordering of rootlets
    segment_order = {segment: i for i, segment in enumerate(SEGMENT_NAMES)}
    side_order = {"left": 0, "right": 1}

    targets = sorted(
        {
            (r["segment"], r["rootlet"], r["side"])
            for r in records
        },
        key=lambda x: (
            segment_order[x[0]],
            x[1],
            side_order[x[2]],
        ),
    )

    # --------------------------------------------------------
    # Use first electrode to define the sensory-fiber layout
    # --------------------------------------------------------

    reference_site = site_ids[0]

    fiber_coordinates = []
    fiber_segments = []
    fiber_rootlets = []
    fiber_sides = []

    reference_coordinates = {}

    for segment, rootlet, side in targets:

        record = next(
            r for r in records
            if r["site"] == reference_site
            and r["segment"] == segment
            and r["rootlet"] == rootlet
            and r["side"] == side
        )

        coords, _ = read_lead_field_file(
            record["filepath"],
            value_column=value_column,
        )

        target = (segment, rootlet, side)
        reference_coordinates[target] = coords

        n_fibers = len(coords)

        fiber_coordinates.extend(coords)
        fiber_segments.extend([segment] * n_fibers)
        fiber_rootlets.extend([rootlet] * n_fibers)
        fiber_sides.extend([side] * n_fibers)

    fiber_coordinates = np.asarray(fiber_coordinates, dtype=float)

    n_sites = len(site_ids)
    n_fibers = len(fiber_coordinates)

    # --------------------------------------------------------
    # LFM: electrode × sensory fiber
    # --------------------------------------------------------

    lfm = np.full(
        (n_sites, n_fibers),
        np.nan,
        dtype=float,
    )

    record_map = { (r["site"], r["segment"], r["rootlet"], r["side"]): r["filepath"] for r in records}

    for site_idx, site in enumerate(
        tqdm(site_ids, desc="Building LFM")
    ):

        rootlet_start = 0

        for segment, rootlet, side in targets:

            filepath = record_map[(site, segment, rootlet, side)]

            coords, potential_v = read_lead_field_file(
                filepath,
                value_column=value_column,
            )

            reference_coords = reference_coordinates[
                (segment, rootlet, side)
            ]

            if coords.shape != reference_coords.shape:
                raise RuntimeError(
                    f"Fiber count mismatch for "
                    f"{segment}_{rootlet}_{side}, site {site}"
                )

            if not np.allclose(coords, reference_coords, atol=1e-6):
                raise RuntimeError(
                    f"Fiber coordinates differ for "
                    f"{segment}_{rootlet}_{side}, site {site}"
                )

            # V -> mV, normalized by reference current in mA
            potential_mv_per_ma = (
                potential_v * 1000.0
                / float(reference_current_ma)
            )

            rootlet_end = rootlet_start + len(potential_mv_per_ma)

            lfm[
                site_idx,
                rootlet_start:rootlet_end,
            ] = potential_mv_per_ma

            rootlet_start = rootlet_end

    if np.isnan(lfm).any():
        raise RuntimeError("LFM contains missing values")

    print("\nLead Field Matrix created")
    print("Electrodes:", n_sites)
    print("Sensory fibers:", n_fibers)
    print("Shape:", lfm.shape)
    print("Units: mV/mA")

    return (
        lfm,
        np.asarray(site_ids),
        fiber_coordinates,
        np.asarray(fiber_segments),
        np.asarray(fiber_rootlets),
        np.asarray(fiber_sides),
    )


# ============================================================
# MuJoCo
# ============================================================

def load_mujoco_model(xml_path):
    xml_path = Path(xml_path)

    if not xml_path.exists():
        raise FileNotFoundError(f"Model XML not found: {xml_path}")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    print("MuJoCo model loaded")
    print("-----------------------------")
    print("nq:", model.nq)
    print("nv:", model.nv)
    print("nu:", model.nu)
    print("na:", model.na)
    print("timestep:", model.opt.timestep)

    return model, data


def get_actuator_names(model):
    names = []
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        names.append(name if name is not None else f"actuator_{i}")
    return names


MYOLEG_GROUP_KEYWORDS = {
    "Iliopsoas": ["iliacus", "psoas"],
    "Adductor": ["addbrev", "addlong", "grac"],
    "Adductor magnus": ["addmag"],
    "Quadriceps": ["recfem", "vaslat", "vasmed", "vasint"],
    "Biceps femoris": ["bflh", "bfsh", "semimem", "semiten"],
    "Gastrocnemius": ["gaslat", "gasmed"],
    "Tibialis anterior": ["tibant"],
    "Extensor hallucis": ["ehl"],
}


def infer_side_from_actuator_name(name):
    if str(name).endswith("_l"):
        return "left"
    if str(name).endswith("_r"):
        return "right"
    return None


def find_group_for_actuator(actuator_name):
    lower_name = str(actuator_name).lower()
    for group, keywords in MYOLEG_GROUP_KEYWORDS.items():
        if any(kw.lower() in lower_name for kw in keywords):
            return group
    return None


def create_actuator_mapping_table(actuator_names):
    rows = []
    for i, actuator in enumerate(actuator_names):
        rows.append(
            {
                "actuator_index": i,
                "actuator": actuator,
                "side": infer_side_from_actuator_name(actuator),
                "muscle_group": find_group_for_actuator(actuator),
            }
        )
    return pd.DataFrame(rows)


def map_muscle_recruitment_to_actuators(
    muscle_recruitment_df,
    actuator_mapping_df,
    missing_value=0.0,
):
    lookup = {
        (row["side"], row["muscle"]): row["muscle_recruitment"]
        for _, row in muscle_recruitment_df.iterrows()
    }

    actuator_recruitment = np.full(
        len(actuator_mapping_df),
        missing_value,
        dtype=float,
    )

    for _, row in actuator_mapping_df.iterrows():
        idx = int(row["actuator_index"])
        side = row["side"]
        group = row["muscle_group"]

        if side is None or group is None:
            continue

        actuator_recruitment[idx] = float(
            lookup.get((side, group), missing_value)
        )

    return np.clip(actuator_recruitment, 0.0, 1.0)


def forward_dynamics(
    activations,
    model,
    data,
    output_folder,
    initial_qpos=None,
    initial_qvel=None,
    dt=None,
    gen_video=False,
    video_name="forward_dynamics.mp4",
    fps=30,
    render_every=None,
    width=640,
    height=480,
    pre_rest_s=5.0,
    video_start_before_activation_s=1.0,
    camera_azimuth=0.0,
    camera_elevation=10.0,
    camera_distance=2.5,
    camera_lookat=(0.0, 0.0, 1.2),
    show_progress=True,
):
    activations = np.asarray(activations, dtype=float)

    if activations.ndim != 2:
        raise ValueError("activations must be T x nu")

    T, nu = activations.shape

    if nu != model.nu:
        raise ValueError(
            f"activations width {nu} does not match model.nu {model.nu}"
        )

    if dt is not None:
        model.opt.timestep = float(dt)

    sim_dt = float(model.opt.timestep)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    mujoco.mj_resetData(model, data)

    if initial_qpos is not None:
        data.qpos[:] = np.asarray(initial_qpos, dtype=float)

    if initial_qvel is not None:
        data.qvel[:] = np.asarray(initial_qvel, dtype=float)

    mujoco.mj_forward(model, data)

    all_qpos = np.zeros((T, model.nq))
    all_qvel = np.zeros((T, model.nv))
    all_ctrl = np.zeros((T, model.nu))
    all_act = np.zeros((T, model.na)) if model.na > 0 else np.zeros((T, 0))

    if render_every is None:
        render_every = max(1, int(round(1.0 / (sim_dt * fps))))

    video_start_step = max(
        0,
        int(
            round(
                (pre_rest_s - video_start_before_activation_s)
                / sim_dt
            )
        ),
    )

    writer = renderer = video_path = None

    if gen_video:
        try:
            import imageio.v2 as imageio_local

            video_path = output_folder / video_name
            writer = imageio_local.get_writer(
                str(video_path),
                fps=fps,
                format="FFMPEG",
            )

            renderer = mujoco.Renderer(
                model,
                height=height,
                width=width,
            )

            scene_option = mujoco.MjvOption()
            scene_option.geomgroup[[1, 3]] = 0
            scene_option.sitegroup[:] = 0

            video_camera = mujoco.MjvCamera()
            video_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            video_camera.azimuth = float(camera_azimuth)
            video_camera.elevation = float(camera_elevation)
            video_camera.distance = float(camera_distance)
            video_camera.lookat[:] = np.asarray(camera_lookat, dtype=float)

        except Exception as e:
            print("Video generation unavailable. Continuing without video.")
            print("Reason:", e)
            gen_video = False
            writer = renderer = video_path = None

    iterator = (
        tqdm(range(T), desc="Forward dynamics")
        if show_progress
        else range(T)
    )

    for idx in iterator:
        ctrl = np.clip(activations[idx, :], 0.0, 1.0)
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)

        all_qpos[idx, :] = data.qpos.copy()
        all_qvel[idx, :] = data.qvel.copy()
        all_ctrl[idx, :] = ctrl

        if model.na > 0:
            all_act[idx, :] = data.act.copy()

        if (
            gen_video
            and idx >= video_start_step
            and idx % render_every == 0
        ):
            renderer.update_scene(
                data,
                camera=video_camera,
                scene_option=scene_option,
            )
            writer.append_data(renderer.render())

    if writer is not None:
        writer.close()

    if renderer is not None:
        renderer.close()

    return all_qpos, all_qvel, all_ctrl, all_act, video_path
