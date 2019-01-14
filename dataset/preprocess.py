import os, sys
import pickle
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


def prefer_pic(ps, p2size):
    if len(ps) == 1: return ps[0]
    best_p = ps[0]
    best_s = p2size[best_p]
    for i in range(1, len(ps)):
        p = ps[i]
        s = p2size[p]
        if s[0] * s[1] > best_s[0] * best_s[1]:  # Select the image with highest resolution
            best_p = p
            best_s = s
    return best_p

def data_cleaning():
  p2size_pth = '/home/data/whale/metadata/p2size.pickle'
  p2h_path = '/home/data/whale/metadata/p2h.pickle'
  tagged = dict([(p, w) for _, p, w in pd.read_csv('/home/data/whale/train.csv').to_records()])

  with open(p2size_pth, 'rb') as f:
    p2size = pickle.load(f)

  with open(p2h_path, 'rb') as f:
    p2h = pickle.load(f)

  h2ps = {}
  for p, h in p2h.items():
    if h not in h2ps: h2ps[h] = []
    if p not in h2ps[h]: h2ps[h].append(p)

  h2p = {}
  for h, ps in h2ps.items():
    h2p[h] = prefer_pic(ps, p2size)

  h2ws = {}
  for p, w in tagged.items():
    if w != 'new_whale':  # Use only identified whales
      h = p2h[p]
      if h not in h2ws: h2ws[h] = []
      if w not in h2ws[h]: h2ws[h].append(w)
  for h, ws in h2ws.items():
    if len(ws) > 1:
      h2ws[h] = sorted(ws)

  # For each whale, find the unambiguous images ids.
  w2hs = {}
  for h, ws in h2ws.items():
    if len(ws) == 1:  # Use only unambiguous pictures
      w = ws[0]
      if w not in w2hs: w2hs[w] = []
      if h not in w2hs[w]: w2hs[w].append(h)
  for w, hs in w2hs.items():
    if len(hs) > 1:
      w2hs[w] = sorted(hs)

  train = []  # A list of training image ids
  for hs in w2hs.values():
    if len(hs) > 1:
      train += hs
  random.shuffle(train)
  train_set = set(train)

  w2ts = {}  # Associate the image ids from train to each whale id.
  for w, hs in w2hs.items():
    for h in hs:
      if h in train_set:
        if w not in w2ts:
          w2ts[w] = []
        if h not in w2ts[w]:
          w2ts[w].append(h)

  return train, w2ts

if __name__ == '__main__':
  train, w2ts = data_cleaning()
  np.save('./annex/train_id.npy', train)
  with open('./annex/w2ts.pickle', 'wb') as f:
    pickle.dump(w2ts, f)
  with open('./annex/w2ts.pickle', 'rb') as f:
    w2ts1 = pickle.load(f)
  train1 = np.load('./annex/train_id.npy')
  print('Successful!')