"""
signal_utils.py
---------------
Signal processing utilities for NILM load fingerprinting.
Configured for REFIT dataset (no-header CSV, 11 columns).

Functions:
    load_refit_house       : Load a REFIT house CSV file
    detect_events          : Detect ON/OFF transitions via edge detection
    extract_event_features : Extract features for a single event window
    build_feature_matrix   : Build full feature matrix from event list
"""

import os
import numpy as np
import pandas as pd
from scipy.signal import medfilt


def load_refit_house(filepath):
    """
    Load a REFIT house CSV file.

    Format: no header, 11 columns:
        Unix, Aggregate, Appliance1..Appliance8, flag

    Missing values marked as -1.

    Returns
    -------
    df     : pd.DataFrame indexed by datetime, columns: mains + Appliance1..8
    labels : dict {column_name: column_name}
    """
    col_names = ['Unix', 'Aggregate', 'Appliance1', 'Appliance2',
                 'Appliance3', 'Appliance4', 'Appliance5', 'Appliance6',
                 'Appliance7', 'Appliance8', 'flag']

    df = pd.read_csv(filepath, header=None, names=col_names)

    # convert unix timestamp to datetime index
    df['Time'] = pd.to_datetime(df['Unix'], unit='s')
    df = df.set_index('Time')

    # drop unneeded columns
    df = df.drop(columns=['Unix', 'flag'])

    # rename Aggregate to mains
    df = df.rename(columns={'Aggregate': 'mains'})

    # -1 means missing in REFIT — replace and forward fill
    df = df.replace(-1, np.nan).ffill().fillna(0)

    # resample to 1s uniform grid
    df = df.resample('1s').mean().ffill().fillna(0)

    labels = {col: col for col in df.columns if col != 'mains'}

    return df, labels


def detect_events(power_series, threshold=50, min_gap_seconds=3):
    """
    Detect state-change events in a power signal.

    Strategy:
      1. Smooth signal with median filter to remove noise
      2. Compute first-order difference
      3. Flag points where |diff| > threshold as events
      4. Suppress duplicates within min_gap_seconds

    Returns pd.DataFrame with columns: timestamp, delta_power, direction
    """
    smoothed = pd.Series(
        medfilt(power_series.values, kernel_size=5),
        index=power_series.index
    )

    diff = smoothed.diff().fillna(0)
    events = []
    last_event_time = None

    for ts, delta in diff.items():
        if abs(delta) >= threshold:
            if last_event_time is None or (ts - last_event_time).seconds >= min_gap_seconds:
                events.append({
                    'timestamp': ts,
                    'delta_power': delta,
                    'direction': 'ON' if delta > 0 else 'OFF'
                })
                last_event_time = ts

    return pd.DataFrame(events)


def extract_event_features(power_series, event_time, window_before=30, window_after=60):
    """
    Extract electrical features for a single detected event.

    Returns dict of features, or None if window is out of range.
    """
    start = event_time - pd.Timedelta(seconds=window_before)
    end   = event_time + pd.Timedelta(seconds=window_after)

    pre  = power_series[start:event_time]
    post = power_series[event_time:end]

    if len(pre) == 0 or len(post) == 0:
        return None

    delta = post.mean() - pre.mean()

    # rise time: seconds to reach 90% of new steady state
    # separates motor loads (slow) from resistive loads (instant)
    target = pre.mean() + 0.9 * delta
    rise_time = window_after
    if delta > 0:
        for ts, val in post.items():
            if val >= target:
                rise_time = (ts - event_time).seconds
                break

    return {
        'delta_power'        : delta,
        'delta_power_abs'    : abs(delta),
        'power_before_mean'  : pre.mean(),
        'power_before_std'   : pre.std(),
        'steady_state_mean'  : post.mean(),
        'steady_state_std'   : post.std(),
        'steady_state_max'   : post.max(),
        'steady_state_min'   : post.min(),
        'rise_time_seconds'  : rise_time,
        'time_of_day_hour'   : event_time.hour,
        'is_nighttime'       : int(event_time.hour >= 22 or event_time.hour <= 6),
    }


def build_feature_matrix(power_series, events_df, labels_series, window_after=60):
    """
    Build full feature matrix from detected events + ground truth labels.

    Returns pd.DataFrame with features + 'label' column.
    """
    rows = []
    for _, event in events_df.iterrows():
        features = extract_event_features(
            power_series, event['timestamp'], window_after=window_after
        )
        if features is None:
            continue

        features['direction'] = 1 if event['direction'] == 'ON' else 0

        try:
            label = labels_series[event['timestamp']]
        except KeyError:
            label = 'unknown'

        features['label'] = label
        rows.append(features)

    return pd.DataFrame(rows)