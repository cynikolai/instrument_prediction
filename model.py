import os
import numpy as np
import scipy.io.wavfile
import xgboost as xgb


# List possible classifications
instruments = ['BassClarinet', 'Cello', 'EbClarinet', 'Marimba', 'TenorTrombone', 'Viola', 'Violin', 'Xylophone']
file_names = {}
path = 'train_data'

# Get data files for each instrument
for instrument in instruments:
    file_names[instrument] = []
    for i in os.listdir(path):
        if os.path.isfile(os.path.join(path,i)) and instrument in i:
            file_names[instrument].append(i)

stride_length = 100
data_arr = []
label_arr = []
data_test_arr = []
label_test_arr = []

i = 0
# Get instrument data
for instrument in instruments:
    for file_name in file_names[instrument]:
        print file_name
        rate, wav_data = scipy.io.wavfile.read(path + '/' + file_name)
        label = instruments.index(instrument)
        strides = (len(wav_data) - 5)/stride_length
        for j in range(strides):
            data = wav_data[j*stride_length:(j+1)*(stride_length)]
            if i < 10:
                data_arr.append(data)
                label_arr.append(label)
                i += 1
            else:
                data_test_arr.append(data)
                label_test_arr.append(label)
                i = 0

# Set up data for training
data_arr = np.asarray(data_arr)
label_arr = np.asarray(label_arr)
data_test_arr = np.asarray(data_test_arr)
label_test_arr = np.asarray(label_test_arr)
dtrain = xgb.DMatrix(data_arr, label=label_arr)
dvalidate = xgb.DMatrix(data_test_arr, label=label_test_arr)

# Train classifier of length 50 sound inputs
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = len(instruments)
num_round = 500
evallist  = [(dtrain,'train'), (dvalidate, 'validate')]
bst = xgb.train(param, dtrain, num_round, evallist)
bst.save_model('instrument_classifier.model')
