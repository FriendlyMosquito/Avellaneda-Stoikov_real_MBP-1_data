from os import path

import pandas as pd
import numpy as np
import json as json
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', 'manifest.json')

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)



## prior data
s = 0 #Estimating sample, size in days
pre_market = {'Var': 0}
market = {'Var': 0}
post_market = {'Var': 0}

for i in range(s):
    path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][i+2]['filename'])
    data = pd.read_csv(path, delimiter=',')

    data['ts_event'] = pd.to_datetime(data['ts_event'])
    cutoff_time = pd.Timestamp('13:30:00').time()
    cutoff_time_post = pd.Timestamp('20:00:00').time()

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
        S = (data['bid_px_00'][n]+data['ask_px_00'][n])/2
        if S_prev is None:
            S_prev = S
            continue
        RV = RV + (S - S_prev) ** 2
        S_prev = S
    post_market['Var'] = post_market['Var'] + RV / T_session

pre_market['Var'] = pre_market['Var'] / s
market['Var'] = market['Var'] / s
post_market['Var'] = post_market['Var'] / s

print(pre_market['Var'])
print(market['Var'])
print(post_market['Var'])
# what was got from s=30, very unnormal result, gotta check the data, thought it was clean
#0.03196793135521876
#309.4396203095001
#0.22286033234953756




#    if data['action'][n] == 'R':

#for i in range(len(manifest['files'])-3):
#    path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][i+2]['filename'])
#    data = pd.read_csv(path, delimiter=',')
#    q = 0
#    X = 0


#    for n in range(len(data)):


