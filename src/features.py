import numpy as np
import pandas as pd


FEATURE_COLUMNS = ["latency", "throughput", "packet_loss"]
LATENCY_SCALE_MAX = 5000.0
THROUGHPUT_SCALE_MAX = 1.0
CONGESTION_SCORE_COLUMN = "congestion_score"


def _numeric_column(df, column):
    """Return a numeric source column with invalid values coerced to NaN."""
    return pd.to_numeric(df[column], errors="coerce")


def _safe_quantile(series, quantile, fallback):
    """Return a positive quantile for scaling, falling back when data is degenerate."""
    value = series.replace([np.inf, -np.inf], np.nan).dropna().quantile(quantile)
    if pd.isna(value) or value <= 0:
        fallback = float(fallback) if not pd.isna(fallback) else 0.0
        return fallback if fallback > 0 else 1.0
    return float(value)


def prepare_model_features(features):
    """Apply the shared train/predict feature preprocessing contract.

    The model sees the same operational ranges exposed by the dashboard sliders:
    latency in 0-5000 ms, throughput in 0-1 where higher means more traffic,
    and packet_loss in 0-1 where higher means worse delivery loss.
    """
    feature_frame = pd.DataFrame(features, columns=FEATURE_COLUMNS).astype(float)
    prepared = pd.DataFrame(index=feature_frame.index)
    prepared["latency"] = feature_frame["latency"].clip(lower=0, upper=LATENCY_SCALE_MAX)
    prepared["throughput"] = feature_frame["throughput"].clip(lower=0, upper=THROUGHPUT_SCALE_MAX)
    prepared["packet_loss"] = feature_frame["packet_loss"].clip(lower=0, upper=1)
    return prepared


def congestion_risk_score(features):
    """Create a monotonic congestion score where each model feature has signal."""
    operational_frame = prepare_model_features(features)
    latency_risk = operational_frame["latency"] / LATENCY_SCALE_MAX
    traffic_pressure = operational_frame["throughput"]
    packet_loss_risk = operational_frame["packet_loss"]
    overload_interaction = np.sqrt(latency_risk * traffic_pressure)
    delivery_interaction = np.sqrt(packet_loss_risk * traffic_pressure)

    return (
        0.30 * latency_risk
        + 0.15 * traffic_pressure
        + 0.40 * packet_loss_risk
        + 0.05 * overload_interaction
        + 0.10 * delivery_interaction
    ).clip(0, 1)


def _build_congestion_target(features):
    """Infer congestion labels from feature risk when no real congestion label exists."""
    congestion_score = congestion_risk_score(features)

    threshold = float(congestion_score.quantile(0.75))
    if pd.isna(threshold) or threshold <= 0:
        threshold = 0.5

    congestion = (congestion_score >= threshold).astype(int)

    if congestion.nunique() < 2:
        threshold = float(congestion_score.median())
        congestion = (congestion_score > threshold).astype(int)

    if congestion.nunique() < 2 and len(congestion) > 1:
        highest_risk = congestion_score.nlargest(max(1, int(len(congestion) * 0.25))).index
        congestion = pd.Series(0, index=features.index)
        congestion.loc[highest_risk] = 1

    return congestion.astype(int), congestion_score


def _label_based_congestion(df, packet_loss):
    """Use CICIDS labels when available, with severe packet loss treated as congestion."""
    label_column = "Label" if "Label" in df.columns else " Label" if " Label" in df.columns else None
    if label_column is None:
        return None

    labels = df[label_column].astype(str).str.strip().str.upper()
    attack_or_anomaly = ~labels.eq("BENIGN")
    severe_packet_loss = packet_loss.astype(float).ge(0.30)
    return (attack_or_anomaly | severe_packet_loss).astype(int)


def _direct_packet_loss(df):
    """Return a direct packet-loss column if the raw data provides one."""
    candidates = [
        "packet_loss",
        "Packet Loss",
        "PacketLoss",
        "Loss Rate",
        "Packet Loss Rate",
        "packet_loss_rate",
    ]
    for column in candidates:
        if column in df.columns:
            values = pd.to_numeric(df[column], errors="coerce")
            if values.dropna().max() > 1:
                values = values / 100
            return values.clip(lower=0, upper=1)
    return None


def _infer_packet_loss(fwd_packets, bwd_packets):
    """Infer delivery-loss risk from bidirectional packet imbalance."""
    fwd_packets = fwd_packets.astype(float)
    bwd_packets = bwd_packets.astype(float)
    forward_deficit = ((fwd_packets - bwd_packets).clip(lower=0) / (fwd_packets + 1)).clip(0, 1)
    confidence_scale = _safe_quantile(fwd_packets, 0.95, fwd_packets.max())
    flow_confidence = (np.log1p(fwd_packets) / np.log1p(confidence_scale)).clip(0, 1)
    return (forward_deficit * flow_confidence).clip(0, 1)


def feature_distribution_report(features):
    """Return distribution diagnostics for model-ready features."""
    report = {}
    for column in FEATURE_COLUMNS:
        series = pd.to_numeric(features[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        report[column] = {
            "min": float(series.min()),
            "p01": float(series.quantile(0.01)),
            "median": float(series.median()),
            "p99": float(series.quantile(0.99)),
            "max": float(series.max()),
            "std": float(series.std(ddof=0)),
            "nunique": int(series.nunique()),
        }
    return report


def validate_feature_frame(features):
    """Fail fast on feature pipeline bugs that make the model ignore inputs."""
    if list(features[FEATURE_COLUMNS].columns) != FEATURE_COLUMNS:
        raise ValueError("Training features must match prediction features exactly.")

    report = feature_distribution_report(features)
    bad_columns = [
        column
        for column, stats in report.items()
        if stats["nunique"] < 2 or stats["std"] <= 1e-9
    ]
    if bad_columns:
        raise ValueError(f"Near-constant model features detected: {bad_columns}")

    out_of_range = []
    if features["latency"].lt(0).any() or features["latency"].gt(LATENCY_SCALE_MAX).any():
        out_of_range.append("latency")
    if features["throughput"].lt(0).any() or features["throughput"].gt(THROUGHPUT_SCALE_MAX).any():
        out_of_range.append("throughput")
    if features["packet_loss"].lt(0).any() or features["packet_loss"].gt(1).any():
        out_of_range.append("packet_loss")
    if out_of_range:
        raise ValueError(f"Model features outside expected ranges: {out_of_range}")

    return report


def _legacy_congestion_score(features):
    """Keep old test helpers aligned with the production target definition."""
    return congestion_risk_score(
        pd.DataFrame(
            {
                "latency": features["latency"],
                "throughput": features["throughput"],
                "packet_loss": features["packet_loss"],
            },
            index=features.index,
        )
    )


def _legacy_congestion_flag(features):
    """Create a compatibility target from the same risk score as training."""
    score = _legacy_congestion_score(features)
    threshold = score.quantile(0.75)
    return (score >= threshold).astype(int)


def _build_legacy_feature_frame(df):
    """Compatibility feature engineering for older tests and notebooks."""
    legacy_df = df.copy()
    if "Flow Duration" not in legacy_df.columns or "Flow Bytes/s" not in legacy_df.columns:
        raise ValueError("Missing required legacy columns")

    packets = pd.to_numeric(
        legacy_df.get("Flow Packets/s", pd.Series(1, index=legacy_df.index)),
        errors="coerce",
    ).replace(0, np.nan)
    legacy_df["latency"] = pd.to_numeric(legacy_df["Flow Duration"], errors="coerce") / packets
    legacy_df["throughput"] = _safe_min_max(pd.to_numeric(legacy_df["Flow Bytes/s"], errors="coerce"))
    legacy_df["packet_loss"] = _safe_min_max(packets)
    return legacy_df.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_COLUMNS)


def _risk_frame(features):
    """Backward-compatible helper for ad hoc diagnostics."""
    prepared = prepare_model_features(features)
    return pd.DataFrame(
        {
            "latency_risk": prepared["latency"] / LATENCY_SCALE_MAX,
            "throughput_risk": prepared["throughput"],
            "packet_loss_risk": prepared["packet_loss"],
        },
        index=features.index,
    )


def create_features(df):
    df = df.copy()

    # Required CICIDS columns
    required_columns = [
        "Flow Duration",
        "Total Fwd Packets",
        "Total Backward Packets",
        "Flow Bytes/s"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    flow_duration = _numeric_column(df, "Flow Duration").clip(lower=0)
    fwd_packets = _numeric_column(df, "Total Fwd Packets").clip(lower=0)
    bwd_packets = _numeric_column(df, "Total Backward Packets").clip(lower=0)
    flow_bytes = _numeric_column(df, "Flow Bytes/s").clip(lower=0)

    total_packets = fwd_packets + bwd_packets
    raw_latency = flow_duration / (total_packets + 1)
    raw_throughput = flow_bytes
    packet_loss = _direct_packet_loss(df)
    if packet_loss is None:
        packet_loss = _infer_packet_loss(fwd_packets, bwd_packets)

    latency_clip = _safe_quantile(raw_latency, 0.99, raw_latency.max())
    throughput_clip = _safe_quantile(raw_throughput, 0.99, raw_throughput.max())

    # Model-ready operational features. These match dashboard prediction inputs:
    # latency 0-5000, throughput 0-1 where higher means more traffic,
    # packet_loss 0-1 where higher always means worse loss.
    df["latency"] = raw_latency.clip(lower=0, upper=latency_clip) / latency_clip * LATENCY_SCALE_MAX
    df["throughput"] = np.log1p(raw_throughput.clip(lower=0, upper=throughput_clip)) / np.log1p(throughput_clip)
    df["packet_loss"] = packet_loss
    df = df.replace([float("inf"), -float("inf")], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS)

    if "congestion" in df.columns:
        df["congestion"] = pd.to_numeric(df["congestion"], errors="coerce").astype("Int64")
    elif "congestion_flag" in df.columns:
        df["congestion"] = pd.to_numeric(df["congestion_flag"], errors="coerce").astype("Int64")
    else:
        label_target = _label_based_congestion(df, df["packet_loss"])
        if label_target is not None:
            df["congestion"] = label_target
        else:
            df["congestion"], df[CONGESTION_SCORE_COLUMN] = _build_congestion_target(df[FEATURE_COLUMNS])

    # Drop invalid rows
    df = df.dropna(subset=FEATURE_COLUMNS + ["congestion"])
    df["congestion"] = df["congestion"].astype(int)

    return df


def _safe_min_max(series):
    """Normalize a series while handling constant values."""
    minimum = series.min()
    maximum = series.max()
    if maximum == minimum:
        return pd.Series(0.0, index=series.index)
    return (series - minimum) / (maximum - minimum)


def _add_legacy_model_features(df):
    """Add backward-compatible engineered columns used by existing tests/code."""
    features = df.copy()
    features["latency_norm"] = _safe_min_max(features["latency"])
    features["throughput_norm"] = _safe_min_max(features["throughput"])
    features["rolling_latency_mean"] = features["latency_norm"].rolling(window=3, min_periods=1).mean()
    features["rolling_throughput_mean"] = features["throughput_norm"].rolling(window=3, min_periods=1).mean()
    features["load_ratio"] = features["latency_norm"] / (features["throughput_norm"] + 1e-6)
    features["packet_intensity"] = features["packet_loss"] * features["latency_norm"]
    features["hour_of_day"] = 0

    if "congestion" in features.columns:
        features["congestion_flag"] = features["congestion"].astype(int)
    elif "congestion_flag" not in features.columns:
        features["congestion_flag"] = _legacy_congestion_flag(features)

    return features


def build_features(df):
    """Backward-compatible wrapper around create_features."""
    try:
        return _add_legacy_model_features(create_features(df))
    except ValueError:
        return _add_legacy_model_features(_build_legacy_feature_frame(df))


def build_prediction_row(latency, throughput, packet_loss):
    """Build a single-row prediction frame with legacy engineered columns."""
    row = pd.DataFrame(
        [
            {
                "latency": latency,
                "throughput": throughput,
                "packet_loss": packet_loss,
            }
        ]
    )
    return _add_legacy_model_features(row)
