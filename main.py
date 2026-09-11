from os import path
from datetime import date
import pandas as pd
import numpy as np
import json as json
import os
import math


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


#variances = var(30, interval='5min') #comment cause running takes too long.
#print(variances)
#print (variances[0], variances[1], variances[2])
vars = [0.019499801432123733, 0.031770039887808396, 0.0386103112427142]

risk = 0.01 #choice between 0 and 1, 1 being no risk, risk defined as the quantity held
k = 1.5 #this needs computing from prior data but hasnt been done( by taking different spread and checking what the fill rate for each would be), for Im taking what the paper used, it kind of goes off the point of simulating fills as Im using real data for that.

def spread(risk, q, var, t, k, market): #market: 0-pre 1-norm 2-post
    T = [('04:00', '09:30', 19800), ('09:30', '16:00', 23400), ('16:00', '20:00', 14400)]
    deltaA = risk * q * var[market] * (T[market][2]-t) + (1/risk) * math.log(1 + risk/k)
    deltaB = -risk * q * var[market] * (T[market][2]-t) + (1/risk) * math.log(1 + risk/k)
    return(deltaA, deltaB)

def prices(deltaA, deltaB, s):
    pA = s + deltaA
    pB = s - deltaB
    return(pA, pB)

def updating(s, e):
    for f in manifest['files'][2:]: # skip condition.json and metadata.json
        cutoff_time = pd.Timestamp('09:30:00').time()
        cutoff_time_post = pd.Timestamp('16:00:00').time()
        fname = f['filename']
        file_date = date(int(fname[10:14]), int(fname[14:16]), int(fname[16:18])) # filenames are equs-mini-YYYYMMDD.mbp-1.csv
        if file_date < s or file_date > e:
            continue

        path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', fname)
        data = pd.read_csv(path, delimiter=',')
        data['ts_event'] = pd.to_datetime(data['ts_event']).dt.tz_convert('America/New_York') #fixes winter/summer times

        p = [None, None]
        q = 0
        X = 0 # allows us to compare each day with the other fairly, no leftover q from before, each day starts with a clean slate

        # for normal market:
        for n in range(len(data)):
            ask_valid = 1
            bid_valid = 1
            if data['ts_event'][n].time() < cutoff_time or data['ts_event'][n].time() >= cutoff_time_post:
                continue
            if data['action'][n] == 'R':
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2:
                continue
            if pd.isna(data['ask_px_00'][n]):
                ask_valid = 0
            if pd.isna(data['bid_px_00'][n]):
                bid_valid = 0
            if ask_valid == 1 and bid_valid == 1:
                s = (data['ask_px_00'][n]+data['bid_px_00'][n])/2
            if p[1] is not None and ask_valid == 1 and data['ask_px_00'][n] < p[1]:
                X -= p[1]
                q += 1
            if p[0] is not None and bid_valid == 1 and data['bid_px_00'][n] > p[0]:
                X += p[0]
                q -= 1

            t = (data['ts_event'][n].hour * 3600 + data['ts_event'][n].minute * 60 + data['ts_event'][n].second) - (cutoff_time.hour * 3600 + cutoff_time.minute * 60 + cutoff_time.second)
            deltas = spread(risk, q, vars, t, k, 1)
            p = prices(deltas[0], deltas[1], s)
            
            
            






s = date(2025, 3, 1)
e = date(2025, 3, 10)
X = 1000000
updating(s, e, X)
