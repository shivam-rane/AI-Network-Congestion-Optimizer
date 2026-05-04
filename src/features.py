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

    # --- Feature Engineering ---
    df["latency"] = df["Flow Duration"] / (
        df["Total Fwd Packets"] + df["Total Backward Packets"] + 1
    )

    df["throughput"] = df["Flow Bytes/s"]

    df["packet_loss"] = 1 - (
        df["Total Backward Packets"] / (df["Total Fwd Packets"] + 1)
    )

    # --- Target Variable (from CICIDS Label) ---
    # Use actual attack labels from dataset instead of derived thresholds
    # This prevents data leakage where target is derived from input features
    if " Label" in df.columns:
        df["congestion"] = (df[" Label"] != "BENIGN").astype(int)
    elif "Label" in df.columns:
        df["congestion"] = (df["Label"] != "BENIGN").astype(int)
    else:
        # Fallback: if no Label column, derive from thresholds (less ideal but functional)
        df["congestion"] = (
            (df["latency"] > df["latency"].quantile(0.75)) |
            (df["packet_loss"] > 0.3)
        ).astype(int)

    # Drop invalid rows
    df = df.replace([float("inf"), -float("inf")], 0)
    df = df.dropna()

    return df
