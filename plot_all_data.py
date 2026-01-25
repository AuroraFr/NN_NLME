import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------
# Global style (tune for LaTeX)
# ---------------------------------------------
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,

    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
})

def preprocess_data(df):
    """
    Pads all sequence data to a unif,rm length for use with jax.lax.scan or jax.vmap.
    """
    grouped = df.groupby("id")
    T_max = grouped.size().max()

    padded_y_list, padded_ts_list, mask_list, injection_times_list = [], [], [], []

    for subject_id, group in grouped:
        group = group.sort_values("Time")
        ts = group["Time"].values

        y = group["ED50_BAU"].values
        second_inj = group.iloc[0]["SecondInjTime"]
        third_inj = group.iloc[0]["ThirdInjTime"]

        Ti = len(ts)
        pad_len = T_max - Ti

        # Pad all sequence data to T_max. Use float for the mask.
        y_padded = np.pad(y.reshape(-1, 1), ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        last_valid_time = ts[-1]
        ts_padded = np.pad(ts, (0, pad_len), 'constant', constant_values=last_valid_time)
        mask = np.array([1.0] * Ti + [0.0] * pad_len)

        padded_y_list.append(y_padded)
        padded_ts_list.append(ts_padded)
        mask_list.append(mask)
        injection_times_list.append(np.array([second_inj, third_inj]))

    # Stack into JAX arrays
    return {
        "padded_y": np.stack(padded_y_list),
        "padded_ts": np.stack(padded_ts_list),
        "mask": np.stack(mask_list),
        "injection_times": np.stack(injection_times_list),
    }

# ---------------------------------------------
# 1) PKPD plot (simu)
# ---------------------------------------------
def plot_pkpd(ax, num_points=5, dataset_id=2, n_subjects=10, seed=0):
    data = np.load(
        f"PKPD_datasets/{num_points}_irregular_1latent_theta1/dataset_{dataset_id}.npy",
        allow_pickle=True
    )
    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, 50, size=n_subjects)

    for i in idxs:
        t = data[i]["t"]
        X1 = data[i]["X1"]
        Y = data[i]["Y"]
        ax.plot(t, X1, alpha=0.7, linewidth=1.2)
        ax.scatter(t, Y, s=18, alpha=0.85)

    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("APV Plasma Concentration (ng/mL)")
    # ax.set_title("Pharmakinetic")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------
# 2) Antibody regular
# ---------------------------------------------
def plot_antibody_regular(ax, measurements=15, duration=400, n_subjects=5):
    data_folder = "antibody_datasets/scipy_antibody_3dose/15_400_50_2latent_theta_F2/"
    dataset = np.load(data_folder + "dataset_0.npy")
    dataset_nonoise = np.load(data_folder + "dataset_nonoise_0.npy")
    timepoints = np.linspace(0, duration, measurements)

    for i in range(n_subjects):
        ax.scatter(timepoints, 10 ** dataset[i, :, 1], s=18, alpha=0.8)
        ax.plot(timepoints, 10 ** dataset_nonoise[i, :, 1], linewidth=1.2)

    ax.axvline(30, color="grey", alpha=0.25, linestyle="--", linewidth=1)
    ax.axvline(250, color="black", alpha=0.6, linestyle="--", linewidth=1)

    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Anti-S IgG (BAU/ml)")
    # ax.set_title("Antibody irregually sampled kinetic")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------
# 3) Antibody irregular
# ---------------------------------------------
def plot_antibody_irregular(ax, n_subjects=5, seed=0):
    data_folder = "antibody_datasets/scipy_antibody_irregular_3dose/10_400_50_2latent_theta_F2/"
    dataset = np.load(data_folder + "dataset_1.npy")
    dataset_nonoise = np.load(data_folder + "dataset_nonoise_1.npy")
    timepoints = np.load(data_folder + "dataset_timepoints_1.npy")

    rng = np.random.default_rng(seed)
    idxs = rng.integers(0, 50, size=n_subjects)

    for i in idxs:
        ax.scatter(timepoints[i], 10 ** dataset[i, :, 1], s=18, alpha=0.8)
        ax.plot(timepoints[i], 10 ** dataset_nonoise[i, :, 1], linewidth=1.2)

    ax.axvline(30, color="grey", alpha=0.25, linestyle="--", linewidth=1)
    ax.axvline(250, color="black", alpha=0.6, linestyle="--", linewidth=1)

    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Anti-S IgG (BAU/ml)")
    # ax.set_title("Antibody irregually sampled kinetic")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------
# 4) Asthma
# ---------------------------------------------
def plot_asthma(ax, n_subjects=10):
    asthme_data = np.load("asthme_datasets/20_400_50_1latent_Kp/dataset_1.npy")
    nonoise_asthme_data = np.load("asthme_datasets/20_400_50_1latent_Kp/dataset_nonoise_1.npy")

    t_eval = np.linspace(0, 400, 20)
    idxs = list(range(n_subjects))

    for i in idxs:
        ax.scatter(t_eval, asthme_data[i, :], s=18, alpha=0.6)
        ax.plot(t_eval, np.log10(nonoise_asthme_data[i, 1, :]), linewidth=1.2)

    ax.set_xlabel("Time (Days)")
    ax.set_ylabel(r"log$_{10}$ $a(t)$ (active TGF-$\beta$ per unit volume)")
    # ax.set_title("Asthma")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------
# 5) Real antibody data (uses your preprocess_data)
# ---------------------------------------------
def plot_real_data(ax, df_path="antibody_datasets/real_ab_data_origin.csv", xlim=(0, 550)):
    # Requires: preprocess_data(df) defined in your codebase
    df = pd.read_csv(df_path)
    df["id"] = df["id"].astype(int)
    df["Time"] = df["Time"].astype(float)
    df["ED50_BAU"] = df["ED50_BAU"].astype(float)
    df = df.dropna()

    data_dict = preprocess_data(df)  # <--- your existing function

    padded_y = np.array(data_dict["padded_y"]).squeeze()
    padded_ts = np.array(data_dict["padded_ts"])
    mask = np.array(data_dict["mask"])
    injection_times = np.array(data_dict["injection_times"])

    num_subjects = padded_y.shape[0]

    # Plot all subjects (semi-transparent)
    for i in range(num_subjects):
        valid_idx = mask[i] == 1.0
        t_plot = padded_ts[i][valid_idx]
        y_plot = padded_y[i][valid_idx]
        ax.plot(t_plot, 10 ** y_plot, marker="o", markersize=2.5, linewidth=1.0)

        inj_2, inj_3 = injection_times[i]
        ax.axvline(inj_2, color="grey", alpha=0.06, linestyle="--", linewidth=1)
        ax.axvline(inj_3, color="black", alpha=0.5, linestyle="--", linewidth=1)

    ax.set_xlim(*xlim)
    ax.set_xlabel("Time (Days)")
    ax.set_ylabel("Anti-S IgG (BAU/ml)")
    # ax.set_title("Real data")
    ax.grid(True, alpha=0.25)


# ---------------------------------------------
# PANEL FIGURE (2x3 same-size panels, 6th empty)
# ---------------------------------------------
def make_5panel_figure(out_pdf="figures/all_5cases_panel.pdf"):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9))
    axes = axes.ravel()

    plot_pkpd(axes[0], num_points=5, dataset_id=2, n_subjects=10, seed=0)
    plot_antibody_regular(axes[1], measurements=15, duration=400, n_subjects=5)
    plot_antibody_irregular(axes[2], n_subjects=5, seed=1)
    plot_asthma(axes[3], n_subjects=10)
    plot_real_data(axes[4], df_path="antibody_datasets/real_ab_data_origin.csv", xlim=(0, 550))

    # 6th panel empty (or use it for a legend/notes)
    import textwrap

    ax = axes[5]
    ax.axis("off")

    items = [
        '(a) Pharmacokinetics (irregular sampling; 6 observations per subject; 0--10 days)',
        '(b) Antibody kinetics (regular sampling; 15 observations per subject; 0--400 days)',
        '(c) Antibody kinetics (irregular sampling; 10 observations per subject; 0--400 days)',
        r"(d) TGF-$\beta$ dynamics (regular sampling; 20 observations per subject; 0--400 days)",
        '(e) Real data (irregular sampling; 2–17 observations per subject)',
    ]

    wrapped = "\n".join(
        textwrap.fill(s, width=48, subsequent_indent="    ")
        for s in items
    )

    ax.text(0.02, 0.92, "Datasets:", transform=ax.transAxes,
            fontsize=16, fontweight="bold", va="top", ha="left", clip_on=False)

    ax.text(-0.05, 0.82, wrapped, transform=ax.transAxes,
            fontsize=16, va="top", ha="left", linespacing=1.5, clip_on=False)

    # Space control (keeps everything same size & readable in LaTeX)
    fig.subplots_adjust(left=0.06, right=0.995, top=0.95, bottom=0.08, wspace=0.25, hspace=0.35)

    # Save: PDF (vector) + PNG (high-res)
    fig.savefig(out_pdf, bbox_inches='tight')
    plt.close(fig)


# ---- run ----
make_5panel_figure()
