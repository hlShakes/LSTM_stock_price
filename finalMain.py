import numpy as npy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

import dataPrepFinal as stk

# Open all pre formatted stock data including: Open, High, Low, Close in that order 
data_file = open("ppb1412.csv", 'r')
# Read all of the lines from the file into memory
data_list = data_file.readlines()
# Close the file (we are done with it)
data_file.close()
#print(data_list)

#for record in data_list:
 #   allRow[x] = record.split(',')
  #  x+=1

datalength = len(data_list)

[training_target,training_data] = stk.dataPrep(189,242,data_list) # call function to define training data 
training_shape = training_data.shape
#print(training_shape)
[data_max, data_min] = stk.maxMin(data_list)
#print(training_target)
# normalise data 
for k in npy.nditer(training_data, op_flags=['readwrite']):
    k[...] = ((k-data_min)/(data_max-data_min))
    pass
for k in npy.nditer(training_target, op_flags=['readwrite']):
    k[...] = ((k-data_min)/(data_max-data_min))
    pass
#print(training_data)

xshape = [training_shape[1],training_shape[2]] 
xshape = npy.array(xshape)

model = Sequential()
model.add(LSTM(64,input_shape=xshape,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64,return_sequences=True))
model.add(LSTM(64))

model.add(Dense(1))
model.compile(loss = 'mean_squared_error',optimizer = 'adam')
model.fit (x = training_data, y= training_target, epochs=5)
training_score = model.evaluate(x = training_data, y= training_target)
print(training_score)

#define test parameters
test_data = stk.colTestData(242,data_list) # call function to define training data 
# normalise data 
for k in npy.nditer(test_data, op_flags=['readwrite']):
    k[...] = ((k-data_min)/(data_max-data_min))
    pass


test_score = model.predict(x = test_data)
print(test_score)
predictedCost = test_score*(data_max-data_min)+data_min

print('Predicted Cost is ', predictedCost)


