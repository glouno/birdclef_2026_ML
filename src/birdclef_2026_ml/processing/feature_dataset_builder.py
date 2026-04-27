import numpy as np
import pandas as pd
from pathlib import Path

from birdclef_2026_ml.configs import PipelineConfig
from birdclef_2026_ml.feature_engineering import extract_features_from_path
from birdclef_2026_ml.feature_engineering.chunking import count_nb_chunks


# TODO: adapt for soundscapes with start_sec, end_sec instead of custom chunking
def build_memmap_from_chunks(
    df: pd.DataFrame,
    pipeline_cfg: PipelineConfig,
    pathroot: Path,
    filename_col: str,
    features_pathroot: Path,
    out_instances_path: Path,
    soundscapes: bool,
    n_classes: int = 234
):

    #  Infer dimensions
    _, feature_names = extract_features_from_path(
        df[filename_col].iloc[0],
        pipeline_cfg=pipeline_cfg,
        pathroot=pathroot,
        features_pathroot=features_pathroot
    )
    n_features = len(feature_names)
    n_instances = count_nb_chunks(df, pipeline_cfg.chunk, pipeline_cfg.feature.sr, pipeline_cfg.feature.hop_length)

    print("n_instances", n_instances)
    out_instances_path.mkdir(parents=True, exist_ok=True)
    X = np.memmap(
        out_instances_path / "X.dat",
        mode="w+",
        dtype=np.float32,
        shape=(n_instances, n_features)
    )

    if soundscapes:
        y = np.memmap(
            out_instances_path / "y.dat",
            mode="w+",
            dtype=np.int8,
            shape=(n_instances, 2, n_classes)  # [:, 0, :] primary label & [:, 1, :] class name
        )
    else:
        y = np.memmap(
            out_instances_path / "y.dat",
            mode="w+",
            dtype=np.int32,
            shape=(n_instances, 2)
        )
    print("Allocated")
    # Fill memmaps
    file_ids = np.empty(n_instances, dtype=object)
    cursor = 0
    for row in df.itertuples(index=False):
        print(f"Cursor {cursor}/{n_instances}")
        filename = getattr(row, filename_col)

        matrix, _ = extract_features_from_path(
            getattr(row, filename_col),
            pipeline_cfg,
            pathroot,
            features_pathroot=features_pathroot
        )

        n_chunks = len(matrix)
        # n_chunks_expect = count_nb_chunks(pd.DataFrame(
        #     data=[row], columns=df.columns), pipeline_cfg.chunk, pipeline_cfg.feature.sr, pipeline_cfg.feature.hop_length)

        # if n_chunks != n_chunks_expect:
        #     print("Wrong estmation", getattr(row, "filename"), getattr(row, "duration"), n_chunks_expect, n_chunks)
        X[cursor:cursor+n_chunks, :] = np.asarray(matrix, dtype=np.float32)
        if soundscapes:
            pl_indices = getattr(row, "primary_label_int_list")
            cn_indices = getattr(row, "class_sname_int_list")

            vec_pl = np.zeros(n_classes, dtype=np.int8)
            vec_cn = np.zeros(n_classes, dtype=np.int8)

            vec_pl[pl_indices] = 1
            vec_cn[cn_indices] = 1

            y[cursor:cursor+n_chunks, 0, :] = vec_cn
            y[cursor:cursor+n_chunks, 1, :] = vec_pl

        else:
            pl = getattr(row, "primary_label_int")
            cn = getattr(row, "class_name_int")

            y[cursor:cursor+n_chunks, 0] = cn
            y[cursor:cursor+n_chunks, 1] = pl

        file_ids[cursor:cursor+n_chunks] = filename
        cursor += n_chunks

    X.flush()
    y.flush()

    # Metadata
    np.save(out_instances_path / "feature_names.npy", feature_names)
    np.save(out_instances_path / "file_ids.npy", file_ids)
    np.save(out_instances_path / "shape_X.npy", np.array(X.shape))
    np.save(out_instances_path / "shape_y.npy", np.array(y.shape))
    np.save(out_instances_path / "dtype_X.npy", np.array(str(X.dtype)))
    np.save(out_instances_path / "dtype_y.npy", np.array(str(y.dtype)))
