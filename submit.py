import os
import time
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pandas import read_csv

import model
from train import score_reshape
from dataset import data_generator

K = tf.keras.backend
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def prepare_submission(threshold, submit, score, known, h2ws, filename):
    """
    Generate a Kaggle submission file.
    @param threshold the score given to ''new_whale''
    @param filename the submission file name
    """
    vtop = 0
    vhigh = 0
    pos = [0, 0, 0, 0, 0, 0]
    with open(filename, 'wt', newline='\n') as f:
        f.write('Image,Id\n')
        for i, p in enumerate(tqdm(submit)):
            t = []
            s = set()
            a = score[i, :]
            for j in list(reversed(np.argsort(a))):
                h = known[j]
                if a[j] < threshold and 'new_whale' not in s:
                    pos[len(t)] += 1
                    s.add('new_whale')
                    t.append('new_whale')
                    if len(t) == 5: break
                for w in h2ws[h]:
                    assert w != 'new_whale'
                    if w not in s:
                        if a[j] > 1.0:
                            vtop += 1
                        elif a[j] >= threshold:
                            vhigh += 1
                        s.add(w)
                        t.append(w)
                        if len(t) == 5: break
                if len(t) == 5: break
            if 'new_whale' not in s: pos[5] += 1
            assert len(t) == 5 and len(s) == 5
            f.write(p + ',' + ' '.join(t[:5]) + '\n')
    return vtop, vhigh, pos


def main():
  config = tf.ConfigProto()
  config.gpu_options.allow_growth=True
  sess = tf.Session(config=config)

  # set session
  K.set_session(sess)
  # Find elements from training sets not 'new_whale'
  tic = time.time()
  # train_df = '/home/data/whale/train.csv'
  # sub_df = '/home/data/whale/sample_submission.csv'
  train_df = '/mnt/home/hdd/hdd1/home/LiaoL/Kaggle/Whale/dataset/train.csv'
  sub_df = '/mnt/home/hdd/hdd1/home/LiaoL/Kaggle/Whale/dataset/sample_submission.csv'
  tagged = dict([(p, w) for _, p, w in read_csv(train_df).to_records()])
  submit = [p for _, p, _ in read_csv(sub_df).to_records()]

  with open('./annex/p2h.pickle', 'rb') as f:
    p2h = pickle.load(f)
  h2ws = {}
  for p, w in tagged.items():
    if w != 'new_whale':  # Use only identified whales
      h = p2h[p]
      if h not in h2ws: h2ws[h] = []
      if w not in h2ws[h]: h2ws[h].append(w)
  known = sorted(list(h2ws.keys()))

  # Dictionary of picture indices
  h2i = {}
  for i, h in enumerate(known): h2i[h] = i
  # Build model and load model weight
  train_model, branch_model, head_model = model.build_model(64e-5, 0.00004)
  tmp = tf.keras.models.load_model('./.checkpoints/model_20-0.63.hdf5')
  train_model.set_weights(tmp.get_weights())
  # Evaluate the model.
  fknown = branch_model.predict_generator(data_generator.FeatureGen(known), max_queue_size=20, workers=10, verbose=0)
  fsubmit = branch_model.predict_generator(data_generator.FeatureGen(submit), max_queue_size=20, workers=10, verbose=0)
  score = head_model.predict_generator(data_generator.ScoreGen(fknown, fsubmit), max_queue_size=20, workers=10, verbose=0)
  score = score_reshape(score, fknown, fsubmit)

  # Generate the subsmission file.
  prepare_submission(0.99, submit, score, known, h2ws, 'submission.csv')
  toc = time.time()
  print("Submission time: ", (toc - tic) / 60.)


if __name__ == '__main__':
    main()
