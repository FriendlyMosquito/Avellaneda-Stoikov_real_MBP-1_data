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
    market = {'Var': 0}

    interval = pd.Timedelta(interval) # min time that must pass before a tick counts as a new sample

    for i in range(s):
        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][i+2]['filename'])
        data = pd.read_csv(path, delimiter=',')

        data['ts_event'] = pd.to_datetime(data['ts_event']).dt.tz_convert('America/New_York') #fixes winter/summer times
        cutoff_time = pd.Timestamp('09:30:00').time()
        cutoff_time_post = pd.Timestamp('16:00:00').time()
    # Found ticks where the ask price is 4000+ which is abnormal and would never fill so ill skip these lines in calculation of Var, by adding a Max Spread allowence

        # Market Open
        RV = 0
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
        market['Var'] = market['Var'] + RV

    market['Var'] = market['Var'] / s
    return(market['Var'])


#print (variances[0], variances[1], variances[2])
#vars = variances = var(30, interval='5min') #comment cause running takes too long.
vars = 24.779591501879146 # in dollars per session
risk = 0.00004

def spread(risk, q, var, k):
    constant = 0.5
    deltaA = (0.5 - q) * risk * var * (constant) + (1/risk) * math.log(1 + risk/k)
    deltaB = (0.5 + q) * risk * var * (constant) + (1/risk) * math.log(1 + risk/k)
    return(deltaA, deltaB)

def spread_symmetric (risk, var, k):
    constant = 0.5
    deltaA = (0.5) * risk * var * (constant) + (1/risk) * math.log(1 + risk/k)
    deltaB = (0.5) * risk * var * (constant) + (1/risk) * math.log(1 + risk/k)
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

def data(dates):
    data_cache = {} # local: returns only the days asked for, no leftovers from earlier calls
    for fname in dates:
        file_date = date(int(fname[10:14]), int(fname[14:16]), int(fname[16:18]))
        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', fname)
        df = pd.read_csv(path, delimiter=',')
        df['ts_event'] = pd.to_datetime(df['ts_event']).dt.tz_convert('America/New_York') #fixes winter/summer times
        df['spread'] = df['ask_px_00'] - df['bid_px_00']
        df['mid'] = (df['ask_px_00'] + df['bid_px_00'])/2
        df['book_ok'] = df['ask_px_00'].notna() & df['bid_px_00'].notna()
        df = df[(df['action'] != 'R') &
        (df['spread'] < 2) &
        (~df['book_ok'] | (df['bid_px_00'] < df['ask_px_00']))].reset_index(drop=True)
        data_cache[file_date] = df
    return data_cache


def updating(data_cache, k, risk=risk, record_trace=False):
    PL = []
    Session_Seconds = 23400
    cutoff_time = pd.Timestamp('09:30:00').time()
    cutoff_time_post = pd.Timestamp('16:00:00').time()
    for file_date, data in data_cache.items():
        p = [None, None]
        q = 0
        X = 0 # allows us to compare each day with the other fairly, no leftover q from before, each day starts with a clean slate
        count = 0 #how many times q changed, just interested to keep track, essentially how many times my chosen prices were hit
        sp = [None, None] # s* = the symmetric-quote book, run side by side on the same ticks as the inventory-aware one
        sq = 0
        sX = 0
        scount = 0
        last_mid = None
        trace = {'t': [], 'mid': [], 'pA': [], 'pB': [], 'q': [], 'X': [],
                 'spA': [], 'spB': [], 'sq': [], 'sX': []} if record_trace else None
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
                if sp[0] is not None and data['side'][n] == 'B' and data['price'][n] >= sp[0]:
                    sX += sp[0]
                    sq -= 1
                    scount += 1
                elif sp[1] is not None and data['side'][n] == 'A' and data['price'][n] <= sp[1]:
                    sX -= sp[1]
                    sq += 1
                    scount += 1
            t = (data['ts_event'][n].hour * 3600 + data['ts_event'][n].minute * 60 + data['ts_event'][n].second) - (cutoff_time.hour * 3600 + cutoff_time.minute * 60 + cutoff_time.second)
            t = t/Session_Seconds

            deltas = spread(risk, q, vars, k)
            symm_deltas = spread_symmetric(risk, vars, k)

            p = prices(deltas[0], deltas[1], data['mid'][n])
            sp = prices(symm_deltas[0], symm_deltas[1], data['mid'][n])
            last_mid = data['mid'][n]
            if record_trace:
                trace['t'].append(data['ts_event'][n])
                trace['mid'].append(data['mid'][n])
                trace['pA'].append(p[0])
                trace['pB'].append(p[1])
                trace['q'].append(q)
                trace['X'].append(X)
                trace['spA'].append(sp[0])
                trace['spB'].append(sp[1])
                trace['sq'].append(sq)
                trace['sX'].append(sX)
        wealth = X + last_mid * q if last_mid is not None else X # mark-to-market: cash plus inventory valued at the day's last mid
        swealth = sX + last_mid * sq if last_mid is not None else sX
        entry = {'date': file_date, 'X': X, 'q': q, 'count': count, 'wealth': wealth,
                 'sX': sX, 'sq': sq, 'scount': scount, 'swealth': swealth}
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
            total_PL.append(sum((entry['wealth']) for entry in PL))
        else:
            total_PL.append(0)
        k_list.append(k_val)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_list, total_PL, marker='o', color='tab:purple')
    ax.set_xlabel('k')
    ax.set_ylabel('Summed wealth')
    ax.set_title('Summed P/L vs k')
    ax.grid(True)
    plt.tight_layout()
    plt.show()
    return k_list, total_PL

#startup check-up, how many ticks per share change in q, to not run bad values and waste time

if __name__ == '__main__':
    s = date(2025, 4, 1)
    e = date(2025, 5, 1)
    k = [10, 15, 20, 25, 30, 35, 40, 45]
    plot_PL_vs_k(s, e, k)

    #plot_PL_vs_k(s, e, [1, 2, 3, 4, 5, 6, 7, 8])
    #PL = updating(s, e, k)
    #plot_PL(PL, s, e)