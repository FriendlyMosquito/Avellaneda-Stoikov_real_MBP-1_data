from os import path
from datetime import date
import pandas as pd
import numpy as np
import json as json
import os
import math
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', 'manifest.json')

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)


def var(s, interval):
    ## prior data
    pre_market = {'Var': 0}
    market = {'Var': 0}
    post_market = {'Var': 0}

    interval = pd.Timedelta(interval) # min time that must pass before a tick counts as a new sample

    for i in range(s):
        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][i+2]['filename'])
        data = pd.read_csv(path, delimiter=',')

        data['ts_event'] = pd.to_datetime(data['ts_event']).dt.tz_convert('America/New_York') #fixes winter/summer times
        cutoff_time = pd.Timestamp('09:30:00').time()
        cutoff_time_post = pd.Timestamp('16:00:00').time()
        data = data[(data['flags'] & 128) != 0].reset_index(drop=True)
    # Found ticks where the ask price is 4000+ which is abnormal and would never fill so ill skip these lines in calculation of Var, by adding a Max Spread allowence

        # Pre Market Open
        RV = 0
        T_session = 19800
        S_prev = None
        last_sample_time = None
        for n in range(len(data)):
            if data['ts_event'][n].time() >= cutoff_time:
                break
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2: # Max Spread Allowence, Honestly picked by what AI said are the percentiles, but it should be higher than this based on simple logic
                continue
            if last_sample_time is not None and data['ts_event'][n] - last_sample_time < interval:
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                last_sample_time = data['ts_event'][n]
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
            last_sample_time = data['ts_event'][n]
        pre_market['Var'] = pre_market['Var'] + RV / T_session

        # Market Open
        RV = 0
        T_session = 23400
        S_prev = None
        last_sample_time = None
        for n in range(len(data)):
            if data['ts_event'][n].time() < cutoff_time or data['ts_event'][n].time() >= cutoff_time_post:
                continue
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2:
                continue
            if last_sample_time is not None and data['ts_event'][n] - last_sample_time < interval:
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                last_sample_time = data['ts_event'][n]
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
            last_sample_time = data['ts_event'][n]
        market['Var'] = market['Var'] + RV / T_session

        # Post Market Open
        RV = 0
        T_session = 14400
        S_prev = None
        last_sample_time = None
        for n in range(len(data)):
            if data['ts_event'][n].time() < cutoff_time_post:
                continue
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2:
                continue
            if last_sample_time is not None and data['ts_event'][n] - last_sample_time < interval:
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                last_sample_time = data['ts_event'][n]
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
            last_sample_time = data['ts_event'][n]
        post_market['Var'] = post_market['Var'] + RV / T_session

    pre_market['Var'] = np.sqrt(pre_market['Var'] / s)
    market['Var'] = np.sqrt(market['Var'] / s)
    post_market['Var'] = np.sqrt(post_market['Var'] / s)
    return(pre_market['Var'], market['Var'], post_market['Var'])


#print (variances[0], variances[1], variances[2])
#vars = variances = var(30, interval='5min') #comment cause running takes too long.
vars = [0.019493815388061855, 0.03176870705369121, 0.0386103112427142]
risk = 0.01 

def spread(risk, q, var, t, k, market): #market: 0-pre 1-norm 2-post
    T = [('04:00', '09:30', 19800), ('09:30', '16:00', 23400), ('16:00', '20:00', 14400)]
    deltaA = (0.5 - q) * risk * var[market]**2 * ((T[market][2]-t)) + (1/risk) * math.log(1 + risk/k)
    deltaB = (0.5 + q) * risk * var[market]**2 * ((T[market][2]-t)) + (1/risk) * math.log(1 + risk/k)
    return(deltaA, deltaB)

def prices(deltaA, deltaB, s):
    pA = s + deltaA
    pB = s - deltaB
    return(pA, pB)

def dates(s, e):
    dates=[]
    for f in manifest['files'][2:]: # skip condition.json and metadata.json
        fname = f['filename']
        file_date = date(int(fname[10:14]), int(fname[14:16]), int(fname[16:18]))
        if file_date < s or file_date > e:
            continue
        else:
            dates.append(fname)
    return dates

data_cache = {}
def data(dates):
    for fname in dates:
        file_date = date(int(fname[10:14]), int(fname[14:16]), int(fname[16:18]))
        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', fname)
        data = pd.read_csv(path, delimiter=',')
        data['ts_event'] = pd.to_datetime(data['ts_event']).dt.tz_convert('America/New_York') #fixes winter/summer times
        data['spread'] = data['ask_px_00'] - data['bid_px_00']
        data['mid'] = (data['ask_px_00'] + data['bid_px_00'])/2
        data['book_ok'] = data['ask_px_00'].notna() & data['bid_px_00'].notna()
        data = data[((data['flags'] & 128) != 0) &
        (data['action'] != 'R') &
        (data['spread'] < 2) &
        (~data['book_ok'] | (data['bid_px_00'] < data['ask_px_00']))].reset_index(drop=True)
        data_cache[file_date] = data
    return data_cache


def updating(data_cache, k, risk=risk, record_trace=False):
    PL = []
    cutoff_time = pd.Timestamp('09:30:00').time()
    cutoff_time_post = pd.Timestamp('16:00:00').time()
    for file_date, data in data_cache.items():
        p = [None, None]
        q = 0
        X = 0 # allows us to compare each day with the other fairly, no leftover q from before, each day starts with a clean slate
        count = 0 #how many times q changed, just interested to keep track, essentially how many times my chosen prices were hit
        last_mid = None
        trace = {'t': [], 'mid': [], 'pA': [], 'pB': [], 'q': [], 'X': []} if record_trace else None
        # for normal market:
        for n in range(len(data)):
            if data['ts_event'][n].time() < cutoff_time or data['ts_event'][n].time() >= cutoff_time_post:
                continue
            if data['action'][n] == 'T':
                if p[0] is not None and data['side'][n] == 'B' and data['price'][n] >= p[0]:
                    X += p[0]
                    q -= 1
                    count += 1
                elif p[1] is not None and data['side'][n] == 'A' and data['price'][n] <= p[1]:
                    X -= p[1]
                    q += 1
                    count += 1
            t = (data['ts_event'][n].hour * 3600 + data['ts_event'][n].minute * 60 + data['ts_event'][n].second) - (cutoff_time.hour * 3600 + cutoff_time.minute * 60 + cutoff_time.second)
            deltas = spread(risk, q, vars, t, k, 1)
            p = prices(deltas[0], deltas[1], data['mid'][n])
            last_mid = data['mid'][n]
            if record_trace:
                trace['t'].append(data['ts_event'][n])
                trace['mid'].append(data['mid'][n])
                trace['pA'].append(p[0])
                trace['pB'].append(p[1])
                trace['q'].append(q)
                trace['X'].append(X)
        wealth = X + last_mid * q if last_mid is not None else X # mark-to-market: cash plus inventory valued at the day's last mid
        entry = {'date': file_date, 'X': X, 'q': q, 'count': count, 'wealth': wealth}
        if record_trace:
            entry['trace'] = trace
        PL.append(entry)
    return PL


def plot_PL_vs_k(s, e, k_values):
    k_list = []
    total_PL = []
    dat = data(dates(s, e))
    for k_val in k_values:
        PL = updating(dat, k_val)
        if sum(entry['count'] for entry in PL) != 0:
            total_PL.append(sum((entry['X']) for entry in PL))
        else:
            total_PL.append(0)
        k_list.append(k_val)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_list, total_PL, marker='o', color='tab:purple')
    ax.set_xlabel('k')
    ax.set_ylabel('Average PL per order fill')
    ax.set_title('Summed P/L vs k')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return k_list, total_PL


if __name__ == '__main__':
    s = date(2025, 3, 1)
    e = date(2025, 3, 5)
    k = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    plot_PL_vs_k(s, e, k)

    #plot_PL_vs_k(s, e, [1, 2, 3, 4, 5, 6, 7, 8])
    #PL = updating(s, e, k)
    #plot_PL(PL, s, e)