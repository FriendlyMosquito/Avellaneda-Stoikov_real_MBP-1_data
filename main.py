from os import path

import pandas as pd
import numpy as np
import json as json
import os


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
        print(i)
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
#Parameters: 0.019499801432123733 0.031770039887808396 0.0386103112427142

risk = 0.01 #choice between 0 and 1, 1 being no risk, risk defined as the quantity held
