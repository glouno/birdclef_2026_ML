from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder


# Type aliases used throughout this module for readability.
Array1D = np.ndarray
Array2D = np.ndarray


@dataclass
class FamilySpeciesHead:
    # Per-family species predictor used in hard approach 1.1.
    # model=None is used when a family has exactly one species.
    model: Any | None
    class_ids: Array1D
    class_prior: Array1D


@dataclass
class HardPerFamilyArtifacts:
    # Trained state for hard approach 1.1.
    family_model: Any
    family_encoder: LabelEncoder
    species_encoder: LabelEncoder
    species_heads: dict[int, FamilySpeciesHead]
    species_to_family_idx: Array1D


@dataclass
class HardMaskedArtifacts:
    # Trained state for hard approach 1.2 (family-conditioned masking).
    family_model: Any
    species_model: Any
    family_encoder: LabelEncoder
    species_encoder: LabelEncoder
    species_to_family_idx: Array1D
    family_species_mask: Array2D


@dataclass
class SoftCombinationArtifacts:
    # Trained state for soft approach.
    family_model: Any
    species_model: Any
    family_encoder: LabelEncoder
    species_encoder: LabelEncoder
    species_to_family_idx: Array1D


def _to_1d(a: Any) -> Array1D:
    """Convert input to 1D ndarray and validate shape."""
    arr = np.asarray(a)
    if arr.ndim != 1:
        raise ValueError("Expected a 1D array-like input")
    return arr


def _validate_same_length(*arrays: Any) -> None:
    """Guard against misaligned training inputs."""
    lengths = [len(x) for x in arrays]
    if len(set(lengths)) != 1:
        raise ValueError(f"Inputs must have same number of rows, got lengths={lengths}")



def _predict_proba_aligned(model: Any, x: Any, all_class_ids: Array1D) -> Array2D:
    """Align predict_proba output to a global class-id ordering.

    Some estimators only return probabilities for classes seen during fit.
    This helper writes those probabilities into a full matrix with stable columns.
    """
    proba = np.asarray(model.predict_proba(x), dtype=float)
    model_classes = np.asarray(model.classes_, dtype=int)

    aligned = np.zeros((proba.shape[0], len(all_class_ids)), dtype=float)
    col_by_class = {int(c): i for i, c in enumerate(all_class_ids)}
    for src_col, cls in enumerate(model_classes):
        aligned[:, col_by_class[int(cls)]] = proba[:, src_col]
    return aligned


def _softmax_with_neg_inf(scores: Array2D) -> Array2D:
    """Softmax that treats -inf as masked classes with zero probability."""
    max_scores = np.max(scores, axis=1, keepdims=True)
    shifted = scores - max_scores
    exp_scores = np.exp(shifted)
    exp_scores[~np.isfinite(scores)] = 0.0
    denom = exp_scores.sum(axis=1, keepdims=True)
    denom = np.where(denom == 0.0, 1.0, denom)
    return exp_scores / denom


def _build_species_to_family_idx(y_family_enc: Array1D, y_species_enc: Array1D, n_species: int) -> Array1D:
    """Build deterministic mapping: species_id -> family_id.

    Raises if a species appears under multiple families.
    """
    species_to_family = np.full(n_species, -1, dtype=int)
    for fam_id, sp_id in zip(y_family_enc, y_species_enc):
        prev = species_to_family[int(sp_id)]
        if prev == -1:
            species_to_family[int(sp_id)] = int(fam_id)
        elif prev != int(fam_id):
            raise ValueError("A species is mapped to multiple families in provided labels")

    if np.any(species_to_family < 0):
        missing = np.where(species_to_family < 0)[0].tolist()
        raise ValueError(f"Some encoded species have no family mapping: {missing}")
    return species_to_family


def _predict_head_proba(head: FamilySpeciesHead, x: Any) -> Array2D:
    """Predict species probabilities for one family-specific head."""
    n_samples = len(x)
    if head.model is None:
        # Single-species family: emit prior (typically [1.0]) for each row.
        return np.tile(head.class_prior, (n_samples, 1))

    proba = np.asarray(head.model.predict_proba(x), dtype=float)
    model_classes = np.asarray(head.model.classes_, dtype=int)

    aligned = np.zeros((proba.shape[0], len(head.class_ids)), dtype=float)
    col_by_class = {int(c): i for i, c in enumerate(head.class_ids)}
    for src_col, cls in enumerate(model_classes):
        aligned[:, col_by_class[int(cls)]] = proba[:, src_col]
    return aligned


def _scores_from_species_model(species_model: Any, x: Any) -> tuple[Array2D, Array1D]:
    """Get comparable class scores from heterogeneous sklearn estimators.

    Preference order: decision_function -> predict_log_proba -> log(predict_proba).
    """
    classes = np.asarray(species_model.classes_, dtype=int)

    if hasattr(species_model, "decision_function"):
        scores = np.asarray(species_model.decision_function(x), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores, classes

    if hasattr(species_model, "predict_log_proba"):
        return np.asarray(species_model.predict_log_proba(x), dtype=float), classes

    proba = np.asarray(species_model.predict_proba(x), dtype=float)
    return np.log(np.clip(proba, 1e-12, 1.0)), classes


def train_hard_per_family_models(
        x: Any,
        y_family: Any,
        y_species: Any,
        family_estimator: Any,
        species_estimator: Any,
) -> HardPerFamilyArtifacts:
    """Hard approach 1.1: train 1 family model + 1 species model per family."""
    y_family = _to_1d(y_family)
    y_species = _to_1d(y_species)
    _validate_same_length(x, y_family, y_species)

    family_encoder = _fit_encoder(y_family)
    species_encoder = _fit_encoder(y_species)

    y_family_enc = np.asarray(family_encoder.transform(y_family), dtype=int)
    y_species_enc = np.asarray(species_encoder.transform(y_species), dtype=int)

    family_model = clone(family_estimator)
    family_model.fit(x, y_family_enc)

    species_heads: dict[int, FamilySpeciesHead] = {}
    for family_id in range(len(family_encoder.classes_)):
        mask = y_family_enc == family_id
        y_sub = y_species_enc[mask]

        class_ids, counts = np.unique(y_sub, return_counts=True)
        class_prior = counts.astype(float) / counts.sum()

        if len(class_ids) == 1:
            # Degenerate case: only one species in this family.
            # We skip model fitting and keep a constant prior head.
            species_heads[family_id] = FamilySpeciesHead(
                model=None,
                class_ids=class_ids.astype(int),
                class_prior=class_prior,
            )
            continue

        model = clone(species_estimator)
        model.fit(np.asarray(x)[mask], y_sub)
        species_heads[family_id] = FamilySpeciesHead(
            model=model,
            class_ids=class_ids.astype(int),
            class_prior=class_prior,
        )

    species_to_family_idx = _build_species_to_family_idx(
        y_family_enc=y_family_enc,
        y_species_enc=y_species_enc,
        n_species=len(species_encoder.classes_),
    )

    return HardPerFamilyArtifacts(
        family_model=family_model,
        family_encoder=family_encoder,
        species_encoder=species_encoder,
        species_heads=species_heads,
        species_to_family_idx=species_to_family_idx,
    )


def predict_proba_hard_per_family(artifacts: HardPerFamilyArtifacts, x: Any) -> Array2D:
    """Hard approach 1.1 inference: route by argmax family, then predict species."""
    n_samples = len(x)
    n_species = len(artifacts.species_encoder.classes_)

    family_ids = np.argmax(artifacts.family_model.predict_proba(x), axis=1)
    out = np.zeros((n_samples, n_species), dtype=float)

    x_arr = np.asarray(x)
    for family_id, head in artifacts.species_heads.items():
        row_mask = family_ids == family_id
        if not np.any(row_mask):
            continue

        chunk_proba = _predict_head_proba(head, x_arr[row_mask])
        out[np.where(row_mask)[0][:, None], head.class_ids[None, :]] = chunk_proba

    return out


def predict_hard_per_family(artifacts: HardPerFamilyArtifacts, x: Any) -> Array1D:
    """Return hard class predictions for approach 1.1."""
    proba = predict_proba_hard_per_family(artifacts, x)
    pred_ids = np.argmax(proba, axis=1)
    return artifacts.species_encoder.inverse_transform(pred_ids)


def train_hard_masked_models(
        x: Any,
        y_family: Any,
        y_species: Any,
        family_estimator: Any,
        species_estimator: Any,
) -> HardMaskedArtifacts:
    """Hard approach 1.2: train family model + global species model."""
    y_family = _to_1d(y_family)
    y_species = _to_1d(y_species)
    _validate_same_length(x, y_family, y_species)

    family_encoder = _fit_encoder(y_family)
    species_encoder = _fit_encoder(y_species)

    y_family_enc = np.asarray(family_encoder.transform(y_family), dtype=int)
    y_species_enc = np.asarray(species_encoder.transform(y_species), dtype=int)

    family_model = clone(family_estimator)
    family_model.fit(x, y_family_enc)

    species_model = clone(species_estimator)
    species_model.fit(x, y_species_enc)

    species_to_family_idx = _build_species_to_family_idx(
        y_family_enc=y_family_enc,
        y_species_enc=y_species_enc,
        n_species=len(species_encoder.classes_),
    )

    n_families = len(family_encoder.classes_)
    n_species = len(species_encoder.classes_)
    # mask[f, s] == True means species s is valid for family f.
    family_species_mask = np.zeros((n_families, n_species), dtype=bool)
    for species_id, family_id in enumerate(species_to_family_idx):
        family_species_mask[int(family_id), int(species_id)] = True

    return HardMaskedArtifacts(
        family_model=family_model,
        species_model=species_model,
        family_encoder=family_encoder,
        species_encoder=species_encoder,
        species_to_family_idx=species_to_family_idx,
        family_species_mask=family_species_mask,
    )


def predict_proba_hard_masked(artifacts: HardMaskedArtifacts, x: Any) -> Array2D:
    """Hard approach 1.2 inference using family-conditioned species masking."""
    family_ids = np.argmax(artifacts.family_model.predict_proba(x), axis=1)

    raw_scores, species_classes = _scores_from_species_model(artifacts.species_model, x)
    n_samples = raw_scores.shape[0]
    n_species = len(artifacts.species_encoder.classes_)

    full_scores = np.full((n_samples, n_species), -np.inf, dtype=float)
    full_scores[:, species_classes] = raw_scores

    for i in range(n_samples):
        family_mask = artifacts.family_species_mask[int(family_ids[i])]
        # Force invalid species logits to -inf before softmax.
        full_scores[i, ~family_mask] = -np.inf

    return _softmax_with_neg_inf(full_scores)


def predict_hard_masked(artifacts: HardMaskedArtifacts, x: Any) -> Array1D:
    """Return hard class predictions for approach 1.2."""
    proba = predict_proba_hard_masked(artifacts, x)
    pred_ids = np.argmax(proba, axis=1)
    return artifacts.species_encoder.inverse_transform(pred_ids)


def train_soft_combination_models(
        x: Any,
        y_family: Any,
        y_species: Any,
        family_estimator: Any,
        species_estimator: Any,
) -> SoftCombinationArtifacts:
    """Soft approach training: train one family model and one species model."""
    y_family = _to_1d(y_family)
    y_species = _to_1d(y_species)
    _validate_same_length(x, y_family, y_species)

    family_encoder = _fit_encoder(y_family)
    species_encoder = _fit_encoder(y_species)

    y_family_enc = np.asarray(family_encoder.transform(y_family), dtype=int)
    y_species_enc = np.asarray(species_encoder.transform(y_species), dtype=int)

    family_model = clone(family_estimator)
    family_model.fit(x, y_family_enc)

    species_model = clone(species_estimator)
    species_model.fit(x, y_species_enc)

    species_to_family_idx = _build_species_to_family_idx(
        y_family_enc=y_family_enc,
        y_species_enc=y_species_enc,
        n_species=len(species_encoder.classes_),
    )

    return SoftCombinationArtifacts(
        family_model=family_model,
        species_model=species_model,
        family_encoder=family_encoder,
        species_encoder=species_encoder,
        species_to_family_idx=species_to_family_idx,
    )


def predict_proba_soft_combination(
        artifacts: SoftCombinationArtifacts,
        x: Any,
        renormalize: bool = True,
) -> Array2D:
    """Soft approach inference.

    Computes:
    P(species|x) <- P(species|x) * P(family_of_species|x)
    """
    n_families = len(artifacts.family_encoder.classes_)
    all_family_ids = np.arange(n_families, dtype=int)
    family_proba = _predict_proba_aligned(artifacts.family_model, x, all_family_ids)

    n_species = len(artifacts.species_encoder.classes_)
    all_species_ids = np.arange(n_species, dtype=int)
    species_proba = _predict_proba_aligned(artifacts.species_model, x, all_species_ids)

    weights = family_proba[:, artifacts.species_to_family_idx]
    combined = species_proba * weights

    if renormalize:
        row_sum = combined.sum(axis=1, keepdims=True)
        row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
        combined = combined / row_sum

    return combined


def predict_soft_combination(artifacts: SoftCombinationArtifacts, x: Any) -> Array1D:
    """Return hard class predictions for soft combination approach."""
    proba = predict_proba_soft_combination(artifacts, x)
    pred_ids = np.argmax(proba, axis=1)
    return artifacts.species_encoder.inverse_transform(pred_ids)
