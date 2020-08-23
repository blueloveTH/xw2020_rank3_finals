import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
import gc

# 场景
scene_mapping = {0:0,10:0,2:0,3:0,7:0,
            9:1,6:1,11:1,14:1,5:1,
           8:2,1:2,4:2,13:2,12:2,
           15:3,16:3,17:3,18:3,19:3}
# 行为
action_mapping = {0:0,9:0,8:0,
           10:1,6:1,1:1,
           2:2,11:2,4:2,
           3:3,14:3,13:3,
           7:4,5:4,12:4,
           15:5,16:6,17:7,18:8,19:9}

kfcv_seed = 44
kfold_func = StratifiedKFold
k = 20

def kfcv_evaluate(model_dir, x, y):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    evals = {'loss':0.0, 'accuracy':0.0}
    index = 0

    for train, val in kfold.split(x, np.argmax(y, axis=-1)):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        
        model = keras.models.load_model('%s/part_%d.h5' % (model_dir, index))

        loss, acc = model.evaluate(x=x[val], y=y[val])
        evals['loss'] += loss / k
        evals['accuracy'] += acc / k
        index += 1
    return evals

def kfcv_predict(model_dir, inputs):
    models = []

    for i in range(k):
        models.append(keras.models.load_model(model_dir + '/part_%d.h5' % i))

    print('%s loaded.' % model_dir)
    result = []
    for m in models:
        result.append(m.predict(inputs))

    print('result got')

    result = sum(result) / len(models)
    return result

def kfcv_fit(builder, x, y,
            epochs, checkpoint_path,
            noise_std=None,
            return_validation_data=False,
            verbose=2,
            batch_size=64,
            extra_callbacks=None,
            initial_fold=0):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)

    #if not isinstance(x, list):
    #    x = [x]

    if checkpoint_path[len(checkpoint_path) - 1] != '/':
        checkpoint_path += '/'

    for i in range(initial_fold, k):
        if os.path.exists(checkpoint_path + 'part_%d.h5' % i):
            os.remove(checkpoint_path + 'part_%d.h5' % i)

    for index, (train, val) in enumerate(kfold.split(x, np.argmax(y, axis=-1))):

        if index < initial_fold:
            index += 1
            continue

        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        model = builder()

        x_train = x[train]
        y_train = y[train]

        if noise_std != None:
            x_train = np.r_[x_train, data_enhance_noise(noise_std, x_train)]
            y_train = np.r_[y_train, y_train]
            x_train, y_train = shuffle(x_train, y_train)
            print('Data enhanced (%s) => %d' % ('noise', len(x_train)))

        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path + 'part_%d.h5' % index,
                                monitor='val_accuracy',
                                verbose=0,
                                mode='max',
                                save_best_only=True)

        callbacks = [checkpoint]
        if extra_callbacks != None:
            for cb in extra_callbacks:
                callbacks.append(cb)

        h = model.fit(x=x_train, y=y_train,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x[val], y[val]),
                callbacks=callbacks,
                batch_size=batch_size,
                shuffle=True
                )

        print('checkpoint.best: %f' % checkpoint.best)

        del model
        gc.collect()

        if return_validation_data:
            return x[val], y[val]

def data_enhance_noise(std, train_data):
    noise = train_data + np.random.normal(0, std, size=train_data.shape)
    return noise

def shuffle(data, labels, seed=None):
    index = [i for i in range(len(labels))]
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(index)

    if isinstance(data, list):
        for i in range(len(data)):
            data[i] = data[i][index]
    else:
        data = data[index]
    return data, labels[index]