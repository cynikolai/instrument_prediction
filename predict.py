import os
import numpy as np
import scipy.io.wavfile
import xgboost as xgb

# List possible classifications
instruments = ['BassClarinet', 'Cello', 'EbClarinet', 'Marimba', 'TenorTrombone', 'Viola', 'Violin', 'Xylophone']
file_names = {}

# Get data for unknown instruments
path = 'test_data'
test_file_names = []
for i in os.listdir(path):
    if os.path.isfile(os.path.join(path,i)):
        test_file_names.append(i)

stride_length = 100

# Load previous model
bst = xgb.Booster({'nthread':4})
bst.load_model("instrument_classifier.model")

# Get classifications for unknown files
output_lines = []
test_data = []
for file_name in test_file_names:
    rate, wav_data = scipy.io.wavfile.read(path + '/' + file_name)
    test_data.append(wav_data[1000:1000+stride_length])
test_data = np.asarray(test_data)
xg_test = xgb.DMatrix(test_data)

pred_prob = bst.predict(xg_test)

for i, file_name in enumerate(test_file_names):
    index_instrument =  int(pred_prob[i])
    instrument = instruments[index_instrument]
    output_lines.append(file_name + "," + instrument)

# Generate output file
with open('cyrus_nikolaidis.csv', 'w') as f:
    for line in output_lines:
        f.write(line)
        f.write('\n')
