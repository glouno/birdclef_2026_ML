import librosa
import numpy as np
import geopandas as gpd
import geodatasets
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch

from birdclef_2026_ml.audio_utils import compute_rms_dbfs, select_silence_frames_from_rms_db
from birdclef_2026_ml.constants import GEO_BBOX_EAST, GEO_BBOX_NORTH, GEO_BBOX_SOUTH, GEO_BBOX_WEST, SAMPLE_RATE


def summarize_df(df):
    def is_list_series(s):
        non_null = s.dropna()
        return len(non_null) > 0 and non_null.map(lambda x: isinstance(x, list)).all()

    summary = pd.DataFrame({
        "dtype": df.dtypes,
        "missing": df.isna().sum(),
        "missing_%": df.isna().mean() * 100,
    })

    summary["n_unique"] = np.nan
    for col in df.columns:
        if not is_list_series(df[col]):
            summary.loc[col, "n_unique"] = df[col].nunique()

    # Add list-specific metrics
    list_cols = [col for col in df.columns if is_list_series(df[col])]

    summary["avg_list_len"] = np.nan
    summary["min_list_len"] = np.nan
    summary["max_list_len"] = np.nan

    for col in list_cols:
        lengths = df[col].dropna().apply(len)
        summary.loc[col, "avg_list_len"] = lengths.mean()
        summary.loc[col, "min_list_len"] = lengths.min()
        summary.loc[col, "max_list_len"] = lengths.max()

    return summary


def plot_audio_overview_on_axes(path, train_audio_dir, axes, n_mfcc=13):
    y, sr = librosa.load(train_audio_dir / path, sr=SAMPLE_RATE)

    stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=1.0)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=1.0)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # db_vmin, db_vmax = -80, 0
    # mfcc_vmin, mfcc_vmax = np.percentile(mfcc, [1, 99])

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')

    img1 = librosa.display.specshow(
        stft_db,
        sr=sr,
        x_axis='time',
        y_axis='hz',
        ax=axes[1],
        cmap='magma',
        # vmin=db_vmin,
        # vmax=db_vmax,
    )
    axes[1].set_title('Spectrogram (dB)')
    fig1 = axes[1].figure
    fig1.colorbar(img1, ax=axes[1], format='%+2.0f dB')

    img2 = librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=axes[2],
        cmap='magma',
        # vmin=db_vmin,
        # vmax=db_vmax,
    )
    axes[2].set_title('Mel Spectrogram (dB)')
    axes[2].set_ylabel('mels')
    fig1.colorbar(img2, ax=axes[2], format='%+2.0f dB')

    img3 = librosa.display.specshow(
        mfcc,
        x_axis='time',
        ax=axes[3],
        cmap='coolwarm',
        # vmin=mfcc_vmin,
        # vmax=mfcc_vmax,
    )
    axes[3].set_title(f'MFCC (n_mfcc={n_mfcc})')
    fig1.colorbar(img3, ax=axes[3])

    return y, sr


def plot_audio_overview(path, train_audio_dir, n_mfcc=13, figsize=(14, 14)):
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=False)
    plot_audio_overview_on_axes(path, train_audio_dir, axes, n_mfcc=n_mfcc)
    plt.tight_layout()


def plot_audio_overview_grid(
    paths_list,
    train_audio_dir,
    n_mfcc=13,
    n_cols=2,
    figsize_per_example=(8, 12),
    extra_titles=None
):
    n_examples = len(paths_list)
    n_rows = int(np.ceil(n_examples / n_cols))

    if extra_titles is not None and len(extra_titles) != n_examples:
        raise ValueError("extra_titles must have the same length as paths_list.")

    # Layout: each example occupies 4 stacked rows in one column.
    fig, axes = plt.subplots(
        n_rows * 4,
        n_cols,
        figsize=(figsize_per_example[0] * n_cols, figsize_per_example[1] * n_rows)
    )
    axes = np.array(axes).reshape(n_rows * 4, n_cols)

    for idx, path in enumerate(paths_list):
        row_block = (idx // n_cols) * 4
        col = idx % n_cols
        panel_axes = axes[row_block:row_block + 4, col]
        plot_audio_overview_on_axes(path, train_audio_dir, panel_axes, n_mfcc=n_mfcc)

        suffix = ""
        if extra_titles is not None and extra_titles[idx] is not None:
            suffix = f"{extra_titles[idx]}"

        panel_axes[0].set_title(f"Waveform: {path}\n{suffix}")

    for idx in range(n_examples, n_rows * n_cols):
        row_block = (idx // n_cols) * 4
        col = idx % n_cols
        for axis in axes[row_block:row_block + 4, col]:
            axis.axis("off")

    plt.tight_layout()


def plot_train_locations(df, hue_col=None, max_categories=20):
    columns = ["latitude", "longitude"]
    if hue_col is not None:
        if hue_col not in df.columns:
            raise ValueError(f"Column '{hue_col}' not found in dataframe.")
        columns.append(hue_col)

    geo = df[columns].dropna(subset=columns)
    if geo.empty:
        raise ValueError("No rows with valid coordinates (and hue values if provided).")

    gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo.longitude, geo.latitude),
        crs="EPSG:4326",
    )

    world = gpd.read_file(geodatasets.get_path("naturalearth.land"))

    print("Train rows with valid coordinates:", len(geo), "/", len(df))
    print("Latitude range:", (geo["latitude"].min(), geo["latitude"].max()))
    print("Longitude range:", (geo["longitude"].min(), geo["longitude"].max()))

    fig, ax = plt.subplots(figsize=(12, 7))
    world.plot(ax=ax, color="lightgray", edgecolor="white")

    if hue_col is None:
        gdf.plot(ax=ax, markersize=3, alpha=0.6, color="steelblue")
    else:
        hue_values = geo[hue_col]

        if pd.api.types.is_numeric_dtype(hue_values):
            print(f"{hue_col} range:", (hue_values.min(), hue_values.max()))
            gdf.plot(
                ax=ax,
                column=hue_col,
                cmap="viridis",
                markersize=3,
                alpha=0.75,
                legend=True,
                legend_kwds={"label": hue_col, "shrink": 0.7},
            )
        else:
            hue_cat = hue_values.astype("string")
            n_categories = int(hue_cat.nunique(dropna=True))
            print(f"{hue_col} categories:", n_categories)

            if n_categories > max_categories:
                top_cats = hue_cat.value_counts().nlargest(max_categories).index
                hue_cat = hue_cat.where(hue_cat.isin(top_cats), other="Other")
                print(
                    f"Too many categories, grouped rare ones into 'Other' "
                    f"(kept top {max_categories})."
                )

            # Order categories so minority classes appear first in the legend.
            cat_counts = hue_cat.value_counts(dropna=True)
            cat_order = cat_counts.sort_values(ascending=True).index.tolist()
            hue_cat = pd.Categorical(hue_cat, categories=cat_order, ordered=True)

            # Draw majority classes first so minority points remain visible on top.
            gdf = gdf.assign(_hue_cat=hue_cat)
            gdf = gdf.assign(_hue_count=pd.Series(hue_cat, index=gdf.index).astype("string").map(cat_counts))
            gdf = gdf.sort_values("_hue_count", ascending=False)
            gdf.plot(
                ax=ax,
                column="_hue_cat",
                categorical=True,
                # cmap="tab20",
                markersize=6,
                alpha=0.5,
                legend=True,
                legend_kwds={"title": hue_col, "loc": "upper left", "bbox_to_anchor": (1.02, 1.0)},
            )

    bbox_left = GEO_BBOX_WEST
    bbox_right = GEO_BBOX_EAST
    bbox_top = GEO_BBOX_NORTH
    bbox_bottom = GEO_BBOX_SOUTH

    bbox = Rectangle(
        (bbox_left, bbox_bottom),
        bbox_right - bbox_left,
        bbox_top - bbox_bottom,
        fill=False,
        edgecolor="crimson",
        linewidth=2,
        linestyle="--",
        label="Recording location (soundscapes)",
    )
    ax.add_patch(bbox)

    title = "Recording locations (train)"
    if hue_col is not None:
        title += f" by {hue_col}"

    plt.title(title)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    # Keep the rectangle legend entry for no-hue and numeric-hue cases.
    if hue_col is None or pd.api.types.is_numeric_dtype(geo[hue_col]):
        plt.legend()

    plt.tight_layout()


def _parse_primary_label_list(value):
    if isinstance(value, list):
        labels = value
    elif pd.isna(value):
        labels = []
    else:
        labels = str(value).split(";")

    return [str(label).strip() for label in labels if str(label).strip()]


def _to_seconds(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    try:
        return float(pd.to_timedelta(str(value)).total_seconds())
    except Exception:
        return np.nan


def plot_soundscape_species_activity(
    soundscapes,
    taxonomy,
    filename=None,
    ax=None,
    show_legend=True,
    title=None,
):
    class_colors = {
        "Aves": "#3B82F6",
        "Mammalia": "#F97316",
        "Amphibia": "#12843C",
        "Reptilia": "#EF4444",
        "Insecta": "#A855F7",
    }

    df = soundscapes.loc[soundscapes["filename"] == filename, ["start", "end", "primary_label_list"]].copy()
    df["start_sec"] = pd.to_timedelta(df["start"]).dt.total_seconds()
    df["duration"] = pd.to_timedelta(df["end"]).dt.total_seconds() - df["start_sec"]

    events = (
        df[["start_sec", "duration", "primary_label_list"]]
        .explode("primary_label_list", ignore_index=True)
        .rename(columns={"primary_label_list": "primary_label"})
    )
    events["primary_label"] = events["primary_label"].astype(str)

    tax = taxonomy[["primary_label", "common_name", "class_name"]].copy()
    tax["primary_label"] = tax["primary_label"].astype(str)
    events = events.merge(tax, on="primary_label", how="left")

    species_order_df = (
        events.groupby(["primary_label", "common_name"], as_index=False)["start_sec"]
        .min()
        .sort_values(["start_sec", "primary_label"])
    )
    species_order = species_order_df["primary_label"].tolist()
    species_labels = [
        f"{row.primary_label} - {row.common_name}"
        for row in species_order_df.itertuples(index=False)
    ]

    if ax is None:
        fig_height = max(1.6, 0.24 * len(species_order) + 0.8)
        _, ax = plt.subplots(figsize=(11, fig_height))

    species_to_y = {label: idx for idx, label in enumerate(species_order)}
    bar_height = 0.52

    for primary_label, grp in events.groupby("primary_label", sort=False):
        y = species_to_y[primary_label] - bar_height / 2
        xranges = list(zip(grp["start_sec"].to_numpy(), grp["duration"].to_numpy()))
        color = class_colors[grp["class_name"].iloc[0]]
        ax.broken_barh(xranges, (y, bar_height), facecolors=color, edgecolors="none", alpha=0.9)

    x_min = float(df["start_sec"].min())
    x_max = float((df["start_sec"] + df["duration"]).max())
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-0.5, len(species_order) - 0.5)
    ax.set_yticks(range(len(species_order)))
    ax.set_yticklabels(species_labels, fontsize=8)

    tick_step = 5 if (x_max - x_min) <= 300 else 30
    ax.set_xticks(np.arange(np.floor(x_min / tick_step) * tick_step, x_max + tick_step, tick_step))

    ax.grid(axis="x", alpha=0.2, linewidth=0.4)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Species")
    ax.set_title(title or f"Species activity timeline - {filename}", fontsize=10)

    if show_legend:
        present_classes = events["class_name"].drop_duplicates().tolist()
        handles = [
            Patch(facecolor=class_colors[class_name], edgecolor="none", label=class_name)
            for class_name in class_colors
            if class_name in present_classes
        ]
        ax.legend(
            handles=handles,
            title="Class",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0,
            fontsize=8,
            title_fontsize=9,
            frameon=True,
        )

    return ax


def plot_waveform_rms_db_with_silence(
    y,
    cfg,
    silence_th=-40.0,
    start_th=None,
    stop_th=None,
    axes=None,
    figsize=(14, 7),
):
    """Plot waveform+normalized RMS (top) and RMS dB (bottom) with silence spans."""
    rms, rms_db, times = compute_rms_dbfs(y, cfg)
    silent_mask, silent_seconds, silence_segments = select_silence_frames_from_rms_db(
        rms_db=rms_db,
        times=times,
        silence_th=silence_th,
        start_th=start_th,
        stop_th=stop_th,
    )

    created_fig = False
    if axes is None:
        fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
        created_fig = True
    else:
        fig = axes[0].figure

    waveform = y / (np.max(np.abs(y)))
    rms_norm = rms / (np.max(rms))
    t_wave = np.arange(len(y)) / cfg.sr

    silence_str = ""
    if start_th and stop_th:
        silence_str = f"[{start_th:.2f}, {stop_th:.2f}] dB"
    else:
        silence_str = f"{silence_th:.2f} dB"

    axes[0].plot(t_wave, waveform, color="steelblue", linewidth=0.8, alpha=0.8, label="Waveform (norm)")
    axes[0].plot(times, rms_norm, color="crimson", linewidth=1.5, label="RMS (norm)")
    axes[0].set_ylabel("Normalized amplitude")
    axes[0].set_title("Waveform + RMS (normalized)")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    axes[1].plot(times, rms_db, color="black", linewidth=1.2, label="RMS (dB)")
    if start_th is None or stop_th is None:
        axes[1].axhline(silence_th, color="red", linestyle="--", linewidth=1.1, label=f"silence_th={silence_th:.1f} dB")
    else:
        axes[1].axhline(start_th, color="red", linestyle="--", linewidth=1.1, label=f"start_th={start_th:.1f} dB")
        axes[1].axhline(stop_th, color="darkorange", linestyle="--", linewidth=1.1, label=f"stop_th={stop_th:.1f} dB")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("dB")
    axes[1].set_title(f"RMS dB with silence threshold = {silence_str}")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.2)

    for start_s, end_s in silence_segments:
        axes[0].axvspan(start_s, end_s, color="yellow", alpha=0.25)
        axes[1].axvspan(start_s, end_s, color="yellow", alpha=0.25)

    if created_fig:
        fig.tight_layout()

    return fig, axes, silent_seconds, silence_segments
