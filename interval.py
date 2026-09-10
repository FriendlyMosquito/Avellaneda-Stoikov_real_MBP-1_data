# i need to find what interval to pick for calculating variance optimaly
# the point is to make sure that noise has less influence
# this code is for testing different intervals and plotting variance for each.
import os
import json

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', 'manifest.json')

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Sampling intervals to sweep over (pandas offset aliases)
INTERVALS = ['1s', '5s', '30s', '1min', '5min', '15min', '30min', '60min']

# name -> (session start, session end, session length in seconds)
SESSIONS = {
    'pre_market': ('04:00', '09:30', 19800),
    'market': ('09:30', '16:00', 23400),
    'post_market': ('16:00', '20:00', 14400),
}


def load_clean_days(s):
    # Read + clean each day once (vectorized), so every interval below reuses it
    # instead of re-reading/re-looping the CSVs per interval.
    days = []
    cols = ['ts_event', 'action', 'bid_px_00', 'ask_px_00']
    for i in range(s):
        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][i + 2]['filename'])
        data = pd.read_csv(path, delimiter=',', usecols=cols)

        data['ts_event'] = pd.to_datetime(data['ts_event']).dt.tz_convert('America/New_York')  # fixes winter/summer times

        # Found ticks where the ask price is 4000+ which is abnormal and would never fill so ill skip these lines in calculation of Var, by adding a Max Spread allowence
        mask = (
            (data['action'] != 'R')
            & data['bid_px_00'].notna()
            & data['ask_px_00'].notna()
            & ((data['ask_px_00'] - data['bid_px_00']) <= 2)  # Max Spread Allowence, Honestly picked by what AI said are the percentiles, but it should be higher than this based on simple logic
        )
        data = data.loc[mask, ['ts_event', 'bid_px_00', 'ask_px_00']]
        data['mid'] = (data['bid_px_00'] + data['ask_px_00']) / 2
        data = data.set_index('ts_event').sort_index()
        data = data[~data.index.duplicated(keep='last')]

        day_date = data.index[0].date() if len(data) else None
        days.append({'mid': data['mid'], 'date': day_date})
    return days


def session_rv(mid, day_date, start_str, end_str, interval):
    # Realized variance for one day/session at a given sampling interval.
    # Samples the mid price on a fixed grid using previous-tick (ffill) interpolation,
    # then sums squared changes between consecutive samples.
    start = pd.Timestamp(f'{day_date} {start_str}', tz=mid.index.tz)
    end = pd.Timestamp(f'{day_date} {end_str}', tz=mid.index.tz)
    grid = pd.date_range(start, end, freq=interval)

    sampled = mid.reindex(grid, method='ffill').dropna()
    if len(sampled) < 2:
        return 0.0
    returns = sampled.diff().dropna()
    return float((returns ** 2).sum())


def var(s, intervals=INTERVALS):
    days = load_clean_days(s)
    results = {name: [] for name in SESSIONS}

    for interval in intervals:
        for name, (start_str, end_str, T_session) in SESSIONS.items():
            acc = 0.0
            n_days = 0
            for day in days:
                if day['date'] is None:
                    continue
                rv = session_rv(day['mid'], day['date'], start_str, end_str, interval)
                acc += rv / T_session
                n_days += 1
            results[name].append(np.sqrt(acc / n_days) if n_days else np.nan)

    return results


def plot_results(results, intervals=INTERVALS):
    interval_seconds = pd.to_timedelta(intervals).total_seconds()

    plt.figure(figsize=(9, 6))
    for name, values in results.items():
        plt.plot(interval_seconds, values, marker='o', label=name)

    plt.xlabel('Sampling interval (seconds)')
    plt.ylabel('Realized volatility')
    plt.title('Realized volatility vs. sampling interval')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'rv_vs_interval.png'))
    plt.show()


if __name__ == '__main__':
    variance = var(30)
    plot_results(variance)

#From the plot for market its vissible that var plateuas at around 5min (300s) while for post and pre market we see that variance just goes down as interval increases, which makes sense as trade intensity goes down, we pick the same price more often
