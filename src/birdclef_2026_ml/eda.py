import librosa
import numpy as np
import geopandas as gpd
import geodatasets
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import ast
import re

from birdclef_2026_ml.constants import GEO_BBOX_EAST, GEO_BBOX_NORTH, GEO_BBOX_SOUTH, GEO_BBOX_WEST


def extract_datetime_from_soundscape_filename(filename):
    match = re.search(r'_(\d{8})_(\d{6})\.ogg$', str(filename))
    if not match:
        return pd.NaT
    dt_str = f"{match.group(1)} {match.group(2)}"
    return pd.to_datetime(dt_str, format='%Y%m%d %H%M%S', errors='coerce')


def train_map_to_nan(df):
    def str_to_list(x):
        return ast.literal_eval(x) if isinstance(x, str) else x

    # String-lists to python lists
    df["secondary_labels"] = df["secondary_labels"].apply(str_to_list)
    df["type"] = df["type"].apply(str_to_list)

    # Filling nans
    df["rating"] = df["rating"].replace(0, np.nan)
    df["author"] = df["author"].replace("Unknown", np.nan)
    df["secondary_labels"] = df["secondary_labels"].apply(lambda x: np.nan if not x else x)
    df["type"] = df["type"].apply(lambda x: np.nan if not x else x)


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
    y, sr = librosa.load(train_audio_dir / path, sr=None)

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


def plot_train_locations(df, hue_col=None):
    columns = ['latitude', 'longitude']
    if hue_col is not None:
        columns.append(hue_col)

    geo = df[columns].dropna(subset=columns)
    gdf = gpd.GeoDataFrame(
        geo,
        geometry=gpd.points_from_xy(geo.longitude, geo.latitude),
        crs='EPSG:4326'
    )

    world = gpd.read_file(geodatasets.get_path('naturalearth.land'))

    print('Train rows with valid coordinates:', len(geo), '/', len(df))
    print('Latitude range:', (geo['latitude'].min(), geo['latitude'].max()))
    print('Longitude range:', (geo['longitude'].min(), geo['longitude'].max()))

    fig, ax = plt.subplots(figsize=(12, 7))
    world.plot(ax=ax, color='lightgray', edgecolor='white')

    if hue_col is None:
        gdf.plot(ax=ax, markersize=3, alpha=0.6, color='steelblue')
    else:
        print(f'{hue_col} range:', (geo[hue_col].min(), geo[hue_col].max()))
        gdf.plot(
            ax=ax,
            column=hue_col,
            cmap='viridis',
            markersize=3,
            alpha=0.75,
            legend=True,
            legend_kwds={'label': hue_col, 'shrink': 0.7}
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
        edgecolor='crimson',
        linewidth=2,
        linestyle='--',
        label='Recording location (soundscapes)'
    )
    ax.add_patch(bbox)

    title = 'Recording locations (train)'
    if hue_col is not None:
        title += f' by {hue_col}'

    plt.title(title)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.legend()
    plt.tight_layout()
