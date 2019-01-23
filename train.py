import os
import random
import pickle
import numpy as np
import tensorflow as tf

import model
from dataset import data_generator

K = tf.keras.backend

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def set_lr(model, lr):
    K.set_value(model.optimizer.lr, float(lr))


def get_lr(model):
    return K.get_value(model.optimizer.lr)


def score_reshape(score, x, y=None):
    """
    Tranformed the packed matrix 'score' into a square matrix.
    @param score the packed matrix
    @param x the first image feature tensor
    @param y the second image feature tensor if different from x
    @result the square matrix
    """
    if y is None:
        # When y is None, score is a packed upper triangular matrix.
        # Unpack, and transpose to form the symmetrical lower triangular matrix.
        m = np.zeros((x.shape[0], x.shape[0]), dtype=K.floatx())
        m[np.triu_indices(x.shape[0], 1)] = score.squeeze()
        m += m.transpose()
    else:
        m = np.zeros((y.shape[0], x.shape[0]), dtype=K.floatx())
        iy, ix = np.indices((y.shape[0], x.shape[0]))
        ix = ix.reshape((ix.size,))
        iy = iy.reshape((iy.size,))
        m[iy, ix] = score.squeeze()
    return m


def main():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True # pylint: disable=E1101
    sess = tf.Session(config=config)

    # set session
    K.set_session(sess)
    train_model, branch_model, head_model = model.build_model(64e-5, 0.0001)
    with open('./annex/w2ts.pickle', 'rb') as f:
        w2ts = pickle.load(f)
    train = np.load('./annex/train_id.npy')
    histories = []
    global_steps = 0

    def make_steps(global_steps, step, ampl):
        """
        Perform training epochs
        @param step Number of epochs to perform
        @param ampl the K, the randomized component of the score matrix.
        """
        # shuffle the training pictures
        random.shuffle(train)

        # Compute the match score for each picture pair
        try:
            features = branch_model.predict_generator(data_generator.FeatureGen(train, verbose=1), 
                                                      max_queue_size=12, workers=6, verbose=0)
            score = head_model.predict_generator(data_generator.ScoreGen(features, verbose=1), 
                                                 max_queue_size=12, workers=6, verbose=0)
            score = score_reshape(score, features)
        except:
            score = np.zeros((len(train), len(train)))
        # score = np.zeros((len(train), len(train)))
        check_cb = tf.keras.callbacks.ModelCheckpoint('./.checkpoints/model_{epoch:02d}-{loss:.2f}.hdf5', monitor='loss', verbose=1)
        def schedule(epoch):
            return 64e-5 * (0.98 ** epoch)
        lr_cb = tf.keras.callbacks.LearningRateScheduler(schedule, verbose=1)
        board_cb = tf.keras.callbacks.TensorBoard(log_dir='./.logs', write_graph=True)
        # Train the model for 'step' epochs
        history = train_model.fit_generator(
            data_generator.TrainingData(train, w2ts,
                                        score + ampl * np.random.random_sample(size=score.shape), 
                                        steps=step, batch_size=64),
            initial_epoch=global_steps, epochs=global_steps + step, 
            callbacks=[check_cb, lr_cb, board_cb],
            max_queue_size=12, workers=6, verbose=1).history
        global_steps += step

        # Collect history data
        history['epochs'] = global_steps
        history['ms'] = np.mean(score)
        history['lr'] = get_lr(train_model)
        print(history['epochs'], history['lr'], history['ms'])
        with open('./.logs/history_%02d.pickle'%(global_steps), 'wb') as f:
            pickle.dump(history, f)
        histories.append(history)
        return global_steps

    if os.path.isfile('./annex/standard.model'):
        tmp = tf.keras.models.load_model('./annex/standard.model')
        train_model.set_weights(tmp.get_weights())
    else:
        # epoch -> 10
        global_steps = make_steps(global_steps, 5, 1000)
        ampl = 100.0
        for _ in range(2):
            print('noise ampl.  = ', ampl)
            global_steps = make_steps(global_steps, 5, ampl)
            ampl = max(1.0, 100 ** -0.1 * ampl)
        # epoch -> 150
        for _ in range(4): 
            global_steps = make_steps(global_steps, 5, 1.0)
        # epoch -> 200
        # set_lr(model, 16e-5)
        for _ in range(4): 
            global_steps = make_steps(global_steps, 5, 0.5)
        # epoch -> 240
        # set_lr(model, 4e-5)
        for _ in range(8): 
            global_steps = make_steps(global_steps, 5, 0.25)
        # epoch -> 250
        # set_lr(model, 1e-5)
        # for _ in range(2): 
        #     global_steps = make_steps(global_steps, 5, 0.25)
        # epoch -> 300
        # weights = train_model.get_weights()
        # train_model, branch_model, head_model = model.build_model(64e-5, 0.0002)
        # train_model.set_weights(weights)
        # for _ in range(10): 
        #     global_steps = make_steps(global_steps, 5, 1.0)
        # # epoch -> 350
        # # set_lr(model, 16e-5)
        # for _ in range(10): 
        #     global_steps = make_steps(global_steps, 5, 0.5)
        # # epoch -> 390
        # # set_lr(model, 4e-5)
        # for _ in range(8): 
        #     global_steps = make_steps(global_steps, 5, 0.25)
        # # epoch -> 400
        # # set_lr(model, 1e-5)
        # for _ in range(2): 
        #     global_steps = make_steps(global_steps, 5, 0.25)
        train_model.save('standard.model')
    train_model.summary()


if __name__ == '__main__':
    main()
