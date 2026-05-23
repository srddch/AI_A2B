import numpy as np
import pandas as pd


def build_rf_feature(site_id, past_4_flows, timestamp, encoder):
    site_encoded = encoder.transform([[site_id]]).flatten()

    flows = np.array(past_4_flows, dtype=float)
    timestamp = pd.to_datetime(timestamp)

    hour = timestamp.hour
    minute = timestamp.minute
    dayofweek = timestamp.dayofweek

    flow_mean = np.mean(flows)
    flow_std = np.std(flows)
    flow_min = np.min(flows)
    flow_max = np.max(flows)
    flow_range = flow_max - flow_min
    flow_last = flows[-1]
    flow_diff = flows[-1] - flows[0]
    flow_trend = np.mean(np.diff(flows))

    hour_decimal = hour + minute / 60

    time_features = np.array([
        np.sin(2 * np.pi * hour_decimal / 24),
        np.cos(2 * np.pi * hour_decimal / 24),
        np.sin(2 * np.pi * dayofweek / 7),
        np.cos(2 * np.pi * dayofweek / 7),
        hour_decimal,
        dayofweek
    ])

    statistical_features = np.array([
        flow_mean,
        flow_std,
        flow_min,
        flow_max,
        flow_range,
        flow_last,
        flow_diff,
        flow_trend
    ])

    return np.concatenate([
        site_encoded,
        flows,
        statistical_features,
        time_features
    ])