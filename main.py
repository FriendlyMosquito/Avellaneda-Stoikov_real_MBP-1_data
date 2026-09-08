from os import path

import pandas as pd
import numpy as np
import json as json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', 'manifest.json')

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)


def var(s):
    ## prior data
    pre_market = {'Var': 0}
    market = {'Var': 0}
    post_market = {'Var': 0}

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
        for n in range(len(data)):
            if data['ts_event'][n].time() >= cutoff_time:
                break
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2: # Max Spread Allowence, Honestly picked by what AI said are the percentiles, but it should be higher than this based on simple logic
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
        pre_market['Var'] = pre_market['Var'] + RV / T_session

        # Market Open 
        RV = 0
        T_session = 23400
        S_prev = None 
        for n in range(len(data)):
            if data['ts_event'][n].time() < cutoff_time or data['ts_event'][n].time() >= cutoff_time_post:
                continue
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2:
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
        market['Var'] = market['Var'] + RV / T_session

        # Post Market Open 
        RV = 0
        T_session = 14400
        S_prev = None 
        for n in range(len(data)):
            if data['ts_event'][n].time() < cutoff_time_post:
                continue
            if data['action'][n] == 'R':
                continue
            if pd.isna(data['ask_px_00'][n]) or pd.isna(data['bid_px_00'][n]):
                continue
            if data['ask_px_00'][n] - data['bid_px_00'][n] > 2:
                continue
            S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
            if S_prev is None:
                S_prev = S
                continue
            RV = RV + (S - S_prev) ** 2
            S_prev = S
        post_market['Var'] = post_market['Var'] + RV / T_session

    pre_market['Var'] = np.sqrt(pre_market['Var'] / s)
    market['Var'] = np.sqrt(market['Var'] / s)
    post_market['Var'] = np.sqrt(post_market['Var'] / s)
    return(pre_market['Var'], market['Var'], post_market['Var'])


#variances = var(30) #comment cause running takes too long.
variances = [0.06251565292204324, 0.08897637394882624, 0.04032049195829899] # what i got after adding in th max spread
