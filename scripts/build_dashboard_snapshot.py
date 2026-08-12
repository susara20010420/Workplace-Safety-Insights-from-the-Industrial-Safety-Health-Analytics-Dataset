#!/usr/bin/env python3
"""Build a single JSON snapshot for the WHSAT live dashboard.

Run this at the end of the notebook pipeline (or from a final Colab cell):

    !python build_dashboard_snapshot.py \
        --project-folder "/content/drive/MyDrive/Work Place Safety Insights" \
        --output "/content/drive/MyDrive/Work Place Safety Insights/dashboard_snapshot.json"

The script never invents missing model results. Missing/incomplete artifacts are
reported in `pipeline.warnings` and relevant dashboard sections are marked unavailable.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
    from sklearn.calibration import calibration_curve
except Exception:  # Dashboard can still export EDA if sklearn is unavailable.
    accuracy_score = confusion_matrix = f1_score = precision_recall_fscore_support = calibration_curve = None


SEVERITY_NAMES = ["Severity 1", "Severity 2", "Severity 3", "Severity 4", "Severity 5"]
ARTIFACTS = [
    "train.csv",
    "test.csv",
    "class_weights.json",
    "transformer_predictions.csv",
    "posterior_summary.csv",
    "bayesian_predictions.csv",
    "bayesian_probs_test.npy",
    "kg_features_train.csv",
    "kg_features_test.csv",
    "fusion_predictions.csv",
    "ensemble_config.json",
]


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if pd.isna(value) if not isinstance(value, (str, bytes)) else False:
        return None
    return value


def read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None


def read_json(path: Path) -> Optional[dict]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def normalize_severity(df: pd.DataFrame) -> pd.Series:
    if "severity" in df.columns:
        s = pd.to_numeric(df["severity"], errors="coerce")
    elif "severity_label" in df.columns:
        s = pd.to_numeric(df["severity_label"], errors="coerce") + 1
    elif "true_severity" in df.columns:
        s = pd.to_numeric(df["true_severity"], errors="coerce")
        if s.dropna().between(0, 4).all():
            s = s + 1
    else:
        return pd.Series(index=df.index, dtype=float)
    return s


def as_class_zero_based(series: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    mask = ~np.isnan(vals)
    if mask.any() and np.nanmin(vals) >= 1 and np.nanmax(vals) <= 5:
        vals = vals - 1
    return vals


def class_report(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> List[dict]:
    if precision_recall_fscore_support is None:
        return []
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(yt, yp, labels=labels, zero_division=0)
    return [
        {
            "class": i + 1,
            "label": SEVERITY_NAMES[i],
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in labels
    ]


def ece_score(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> Optional[float]:
    if probs is None or probs.size == 0:
        return None
    labels = labels.astype(int)
    conf = np.max(probs, axis=1)
    pred = np.argmax(probs, axis=1)
    correct = pred == labels
    bins = np.linspace(0, 1, n_bins + 1)
    score = 0.0
    for i in range(n_bins):
        idx = (conf > bins[i]) & (conf <= bins[i + 1])
        if idx.any():
            score += abs(correct[idx].mean() - conf[idx].mean()) * idx.mean()
    return float(score)


def calibration_points(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> List[dict]:
    if calibration_curve is None or probs is None:
        return []
    out = []
    for i in range(min(probs.shape[1], 5)):
        try:
            frac, mean = calibration_curve((labels == i).astype(int), probs[:, i], n_bins=n_bins)
            out.append({
                "class": i + 1,
                "points": [{"predicted": float(x), "observed": float(y)} for x, y in zip(mean, frac)],
            })
        except Exception:
            continue
    return out


def confidence_hist(conf: pd.Series) -> List[dict]:
    c = pd.to_numeric(conf, errors="coerce").dropna().clip(0, 1)
    if c.empty:
        return []
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(c.to_numpy(), bins=edges)
    return [
        {"label": f"{edges[i]:.1f}–{edges[i+1]:.1f}", "count": int(counts[i])}
        for i in range(len(counts))
    ]


def probability_matrix(df: pd.DataFrame, prefixes: List[str]) -> Optional[np.ndarray]:
    for prefix in prefixes:
        cols = [f"{prefix}{i}" for i in range(5)]
        if all(c in df.columns for c in cols):
            arr = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if np.isfinite(arr).all():
                return arr
    return None


def prediction_section(df: Optional[pd.DataFrame], kind: str) -> dict:
    if df is None or df.empty:
        return {"available": False, "kind": kind}

    true_col = "true_severity" if "true_severity" in df.columns else "severity_label" if "severity_label" in df.columns else None
    pred_col = "predicted_severity" if "predicted_severity" in df.columns else None
    if not true_col or not pred_col:
        return {"available": False, "kind": kind, "reason": "true/predicted severity columns missing"}

    y_true = as_class_zero_based(df[true_col])
    y_pred = as_class_zero_based(df[pred_col])
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    yt = y_true[mask].astype(int)
    yp = y_pred[mask].astype(int)

    metrics = {}
    if len(yt) and accuracy_score is not None:
        metrics = {
            "accuracy": float(accuracy_score(yt, yp)),
            "macro_f1": float(f1_score(yt, yp, average="macro")),
            "n": int(len(yt)),
        }

    prefixes = ["prob_class_", "prob_ensemble_", "prob_bayesian_"]
    if kind == "fusion":
        prefixes = ["prob_ensemble_", "prob_class_", "prob_bayesian_"]
    elif kind == "bayesian":
        prefixes = ["prob_bayesian_", "prob_class_", "prob_ensemble_"]

    probs = probability_matrix(df.loc[mask].reset_index(drop=True), prefixes)
    conf = (
        pd.to_numeric(df.loc[mask, "confidence"], errors="coerce")
        if "confidence" in df.columns
        else pd.Series(np.max(probs, axis=1) if probs is not None else [], dtype=float)
    )

    section = {
        "available": True,
        "kind": kind,
        "metrics": metrics,
        "class_report": class_report(y_true, y_pred),
        "confusion_matrix": confusion_matrix(yt, yp, labels=list(range(5))).tolist() if confusion_matrix is not None else [],
        "avg_confidence": float(conf.mean()) if len(conf.dropna()) else None,
        "high_uncertainty_count": int((conf < 0.60).sum()) if len(conf) else None,
        "confidence_histogram": confidence_hist(conf),
    }
    if probs is not None and len(yt) == len(probs):
        section["ece"] = ece_score(probs, yt)
        section["calibration"] = calibration_points(probs, yt)
    else:
        section["ece"] = None
        section["calibration"] = []
    return section


def build_eda(all_df: Optional[pd.DataFrame]) -> dict:
    if all_df is None or all_df.empty:
        return {"available": False}
    df = all_df.copy()
    sev = normalize_severity(df)
    df["_severity"] = sev

    counts = sev.value_counts().sort_index()
    severity_distribution = [
        {"severity": int(k), "label": f"Severity {int(k)}", "count": int(v)}
        for k, v in counts.items() if pd.notna(k)
    ]

    gap_by_severity = []
    if "near_miss_gap" in df.columns:
        g = df.groupby("_severity", dropna=True)["near_miss_gap"].mean()
        gap_by_severity = [
            {"severity": int(k), "label": f"Severity {int(k)}", "mean_gap": float(v)}
            for k, v in g.items() if pd.notna(k) and pd.notna(v)
        ]

    monthly = []
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        valid = dates.dropna()
        if not valid.empty:
            vc = valid.dt.to_period("M").astype(str).value_counts().sort_index()
            monthly = [{"month": str(k), "count": int(v)} for k, v in vc.items()]

    sector_counts = []
    sector_gap = []
    sector_col = "Industry Sector" if "Industry Sector" in df.columns else None
    if sector_col:
        sc = df[sector_col].fillna("Unknown").astype(str).value_counts().head(12)
        sector_counts = [{"sector": str(k), "count": int(v)} for k, v in sc.items()]
        if "near_miss_gap" in df.columns:
            sg = df.groupby(sector_col, dropna=False)["near_miss_gap"].mean().sort_values(ascending=False).head(12)
            sector_gap = [{"sector": "Unknown" if pd.isna(k) else str(k), "mean_gap": float(v)} for k, v in sg.items() if pd.notna(v)]

    critical_heatmap = {"risks": [], "severities": [1, 2, 3, 4, 5], "matrix": []}
    if "critical_risk" in df.columns:
        top = df["critical_risk"].fillna("Unknown").astype(str).value_counts().head(10).index
        tmp = df[df["critical_risk"].fillna("Unknown").astype(str).isin(top)].copy()
        tmp["_risk"] = tmp["critical_risk"].fillna("Unknown").astype(str)
        ct = pd.crosstab(tmp["_risk"], tmp["_severity"]).reindex(index=top, columns=[1, 2, 3, 4, 5], fill_value=0)
        critical_heatmap = {
            "risks": [str(x) for x in ct.index],
            "severities": [1, 2, 3, 4, 5],
            "matrix": ct.astype(int).values.tolist(),
        }

    gender_severity = []
    if "gender" in df.columns:
        ct = pd.crosstab(df["gender"].fillna("Unknown"), df["_severity"]).reindex(columns=[1,2,3,4,5], fill_value=0)
        for gender, row in ct.iterrows():
            for sev_level in [1,2,3,4,5]:
                gender_severity.append({"gender": str(gender), "severity": sev_level, "count": int(row.get(sev_level, 0))})

    high = int((sev >= 4).sum())
    avg_gap = float(pd.to_numeric(df.get("near_miss_gap"), errors="coerce").mean()) if "near_miss_gap" in df.columns else None

    return {
        "available": True,
        "total_incidents": int(len(df)),
        "high_severity_count": high,
        "high_severity_pct": float(high / len(df)) if len(df) else None,
        "avg_near_miss_gap": avg_gap,
        "severity_distribution": severity_distribution,
        "gap_by_severity": gap_by_severity,
        "monthly_trend": monthly,
        "sector_counts": sector_counts,
        "sector_gap": sector_gap,
        "critical_risk_heatmap": critical_heatmap,
        "gender_severity": gender_severity,
    }


def kg_section(train_kg: Optional[pd.DataFrame], test_kg: Optional[pd.DataFrame]) -> dict:
    frames = [x for x in [train_kg, test_kg] if x is not None and not x.empty]
    if not frames:
        return {"available": False}
    df = pd.concat(frames, ignore_index=True, sort=False)
    count_cols = [c for c in df.columns if c.startswith("count_")]
    violation_cols = [c for c in df.columns if c.endswith("_violation")]

    entities = []
    for c in count_cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        entities.append({"entity_type": c.replace("count_", ""), "total": float(vals.sum()), "mean_per_incident": float(vals.mean())})
    entities.sort(key=lambda x: x["total"], reverse=True)

    violations = []
    for c in violation_cols:
        vals = pd.to_numeric(df[c], errors="coerce").fillna(0)
        violations.append({"rule": c.replace("_", " ").title(), "count": int((vals > 0).sum()), "rate": float((vals > 0).mean())})

    control_col = "count_Control"
    coverage = None
    if control_col in df.columns:
        vals = pd.to_numeric(df[control_col], errors="coerce").fillna(0)
        coverage = float((vals > 0).mean())

    return {
        "available": True,
        "processed_incidents": int(len(df)),
        "train_rows": int(len(train_kg)) if train_kg is not None else 0,
        "test_rows": int(len(test_kg)) if test_kg is not None else 0,
        "control_entity_coverage": coverage,
        "entity_counts": entities,
        "violations": violations,
    }


def posterior_section(df: Optional[pd.DataFrame]) -> dict:
    if df is None or df.empty:
        return {"available": False}
    work = df.copy()
    if work.index.name is not None or work.columns[0].startswith("Unnamed"):
        work = work.reset_index(drop=False)
    cols = [c for c in ["mean", "sd", "hdi_2.5%", "hdi_97.5%", "r_hat", "ess_bulk"] if c in work.columns]
    first_col = work.columns[0] if len(work.columns) else None
    preview_cols = ([first_col] if first_col is not None else []) + [c for c in cols if c != first_col]
    preview = work[preview_cols].head(30).replace({np.nan: None}).to_dict(orient="records") if preview_cols else []
    return {"available": True, "rows": int(len(work)), "preview": preview}


def build_snapshot(project_folder: Path) -> dict:
    files = {name: project_folder / name for name in ARTIFACTS}
    status = []
    for name, path in files.items():
        status.append({
            "name": name,
            "present": path.exists(),
            "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat() if path.exists() else None,
            "size_bytes": path.stat().st_size if path.exists() else None,
        })

    train = read_csv(files["train.csv"])
    test = read_csv(files["test.csv"])
    all_df = None
    if train is not None or test is not None:
        all_df = pd.concat([x for x in [train, test] if x is not None], ignore_index=True, sort=False)

    transformer_df = read_csv(files["transformer_predictions.csv"])
    bayesian_df = read_csv(files["bayesian_predictions.csv"])
    fusion_df = read_csv(files["fusion_predictions.csv"])
    kg_train = read_csv(files["kg_features_train.csv"])
    kg_test = read_csv(files["kg_features_test.csv"])
    posterior = read_csv(files["posterior_summary.csv"])
    class_weights = read_json(files["class_weights.json"])
    ensemble_config = read_json(files["ensemble_config.json"])

    eda = build_eda(all_df)
    transformer = prediction_section(transformer_df, "transformer")
    bayesian = prediction_section(bayesian_df, "bayesian")
    fusion = prediction_section(fusion_df, "fusion")
    kg = kg_section(kg_train, kg_test)

    warnings: List[str] = []
    readiness = {
        "eda": train is not None and test is not None,
        "transformer": transformer_df is not None,
        "bayesian_probs": files["bayesian_probs_test.npy"].exists(),
        "kg_train": kg_train is not None,
        "kg_test": kg_test is not None,
        "fusion_predictions": fusion_df is not None,
    }

    if not readiness["bayesian_probs"]:
        warnings.append("bayesian_probs_test.npy is missing. The current fusion notebook is designed to fall back to Transformer probabilities when this file is absent.")
    if not readiness["kg_test"]:
        warnings.append("kg_features_test.csv is missing. The current fusion notebook creates dummy KG test features when this file is absent.")
    if kg_train is not None and len(kg_train) < (len(train) if train is not None else len(kg_train)):
        warnings.append(f"KG train features cover {len(kg_train)} rows, fewer than the available training rows. KG metrics therefore describe only the processed subset.")

    bayes_copy_detected = False
    if transformer_df is not None and bayesian_df is not None:
        common = [c for c in ["predicted_severity", "confidence"] + [f"prob_class_{i}" for i in range(5)] if c in transformer_df.columns and c in bayesian_df.columns]
        if common and len(transformer_df) == len(bayesian_df):
            try:
                bayes_copy_detected = transformer_df[common].reset_index(drop=True).equals(bayesian_df[common].reset_index(drop=True))
            except Exception:
                pass
    if bayes_copy_detected:
        warnings.append("bayesian_predictions.csv matches the Transformer prediction columns. Treat it as a placeholder, not independent Bayesian predictions.")
        bayesian["independent_model_output"] = False
    elif bayesian.get("available"):
        bayesian["independent_model_output"] = True

    ensemble_description = str((ensemble_config or {}).get("description", ""))
    ensemble_description_l = ensemble_description.lower()
    fusion_uses_kg = ("knowledge graph" in ensemble_description_l) or (" kg" in f" {ensemble_description_l}")

    # `fusion_predictions.csv` in the supplied fusion notebook is a weighted average of
    # Transformer and Bayesian probabilities. Treat that saved artifact as a two-model
    # ensemble unless its configuration explicitly states that KG features are included.
    fusion_base_valid = bool(readiness["fusion_predictions"] and readiness["bayesian_probs"] and not bayes_copy_detected)
    full_three_model_valid = bool(fusion_base_valid and readiness["kg_test"] and fusion_uses_kg)
    fusion["pipeline_valid"] = fusion_base_valid
    fusion["full_three_model_valid"] = full_three_model_valid
    fusion["uses_kg_in_saved_predictions"] = fusion_uses_kg
    fusion["ensemble_description"] = ensemble_description or None

    if fusion.get("available") and not fusion_base_valid:
        fusion["label"] = "Saved fusion output present, but independent Bayesian inputs are incomplete"
    if fusion.get("available") and not fusion_uses_kg:
        warnings.append("The saved fusion_predictions.csv is configured as a weighted Transformer + Bayesian ensemble; KG features are not represented in that saved final prediction artifact.")
    if not full_three_model_valid:
        warnings.append("Do not label the current saved fusion predictions as a validated Transformer + Bayesian + KG ensemble. A real KG test feature path and a leakage-free fusion/meta-model training procedure are still required for that claim.")

    overview = {
        "total_incidents": eda.get("total_incidents"),
        "high_severity_count": eda.get("high_severity_count"),
        "high_severity_pct": eda.get("high_severity_pct"),
        "avg_near_miss_gap": eda.get("avg_near_miss_gap"),
        "transformer_macro_f1": transformer.get("metrics", {}).get("macro_f1") if transformer.get("available") else None,
        "transformer_accuracy": transformer.get("metrics", {}).get("accuracy") if transformer.get("available") else None,
        "avg_confidence": transformer.get("avg_confidence") if transformer.get("available") else None,
        "uncertain_cases": transformer.get("high_uncertainty_count") if transformer.get("available") else None,
        "kg_processed_incidents": kg.get("processed_incidents") if kg.get("available") else None,
        "fusion_ready": fusion_base_valid,
        "full_three_model_fusion_ready": full_three_model_valid,
    }

    return json_safe({
        "meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "project_folder": str(project_folder),
            "schema_version": "1.0",
        },
        "overview": overview,
        "eda": eda,
        "transformer": transformer,
        "bayesian": bayesian,
        "posterior": posterior_section(posterior),
        "kg": kg,
        "fusion": fusion,
        "class_weights": class_weights or {},
        "ensemble_config": ensemble_config or {},
        "pipeline": {
            "artifacts": status,
            "readiness": readiness,
            "warnings": warnings,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-folder", required=True, help="Folder containing notebook output artifacts")
    parser.add_argument("--output", required=True, help="Path for dashboard_snapshot.json")
    args = parser.parse_args()

    project_folder = Path(args.project_folder)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot = build_snapshot(project_folder)
    with out.open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False, allow_nan=False)
    print(f"Dashboard snapshot written to: {out}")
    print(f"Warnings: {len(snapshot['pipeline']['warnings'])}")
    for warning in snapshot["pipeline"]["warnings"]:
        print(" -", warning)


if __name__ == "__main__":
    main()
