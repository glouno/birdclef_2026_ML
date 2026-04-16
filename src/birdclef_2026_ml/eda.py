import librosa
import numpy as np
import geopandas as gpd
import geodatasets
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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

    stft_db = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    mel_db = librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr), ref=np.max)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

    librosa.display.waveshow(y, sr=sr, ax=axes[0])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Amplitude')

    img1 = librosa.display.specshow(stft_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
    axes[1].set_title('Spectrogram (dB)')
    fig1 = axes[1].figure
    fig1.colorbar(img1, ax=axes[1], format='%+2.0f dB')

    img2 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
    axes[2].set_title('Mel Spectrogram (dB)')
    axes[2].set_ylabel('mels')
    fig1.colorbar(img2, ax=axes[2], format='%+2.0f dB')

    img3 = librosa.display.specshow(mfcc, x_axis='time', ax=axes[3])
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
