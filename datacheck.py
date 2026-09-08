from os import path
from collections import Counter

import pandas as pd
import numpy as np
import json as json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MANIFEST_PATH = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', 'manifest.json')

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][2]['filename'])
data = pd.read_csv(path, delimiter=',')
unique = data['flags'].value_counts()
n=[]
s=0 # was 30
for j in range(s):
    path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][j+2]['filename'])
    data = pd.read_csv(path, delimiter=',')
    for i in range(len(data)):
        if data['bid_px_00'][i] > data['ask_px_00'][i]:
            n.append(data['publisher_id'][i])
m = Counter(n)
print(m) # all the crossed data are from publisher_id 95

# as this must be a reporting issue, or ill atleast assume it is, ill drop any crossed data from the dataset, first checking all the dates not only the 30 first days.

# my bad code that's slow
for j in range(1):
    path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', manifest['files'][j+2]['filename'])
    data = pd.read_csv(path, delimiter=',')
    for i in range(len(data)):
        if data['bid_px_00'][i] > data['ask_px_00'][i]:
            n.append(data['publisher_id'][i])
            print(j, len(manifest['files']))
m = Counter(n)
print(m)

# ai improved the speed. I need to learn hpow to vectorize myself
n = []
for j in range(len(manifest['files']) - 2):
    fname = manifest['files'][j+2]['filename']
    path = os.path.join(BASE_DIR, 'Data', 'MSFT_MBP-1_CSV', fname)
    data = pd.read_csv(path, usecols=['bid_px_00', 'ask_px_00', 'publisher_id'])

    crossed = data.loc[data['bid_px_00'] > data['ask_px_00'], 'publisher_id']
    n.extend(crossed.tolist())

    print(j, len(manifest['files']))  # once per file, not per row

m = Counter(n)
print(m) # the conclusion is the same, all the crossed data are from publisher_id 95, so ill drop them from the dataset.

# Lets fix the data, found spots where the ask price is 4000+ which is bonkers

