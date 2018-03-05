import socket
import sys
import csv
import numpy
import os
import glob
import random
import numpy as np
import time
from datetime import datetime

from keras.models import Sequential
from keras.models import model_from_json

import matplotlib.pyplot as plt

# Get a random file from the dataset folder
files = glob.glob("./data/*.txt")
file = random.choice(files)

print("File: " + file)

# Read and split the file
data = np.array(list(csv.reader(open(file, "r"), delimiter=";", quoting=csv.QUOTE_NONNUMERIC)))
    

# Remove some features to use the same of the train
data_p1 = data[:, 1:6]				# Throttle, brake, steering, handbrake, speed
data_p2 = data[:, 10:46-18]			# lidar from 0 to 180 degres

# Append the input features (23 features)
input_data = np.append(data_p1, data_p2 , axis=1)

print("Input data: " + str(input_data.shape))

# Load json and create model
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights int the new model
loaded_model.load_weights("./model/model.h5")
loaded_model.summary()

prediction = []

i = 0
sum_time = 0
for input_ in input_data:

	# Predict something	
	start_time = int(round(time.time() * 1000))
	pred = loaded_model.predict(input_.reshape(1,23))
	end_time = int(round(time.time() * 1000))
	
	# Convert matrix to array?? [[]] -> []
	prediction.append(pred[0])
	print(pred[0])

	sum_time += end_time-start_time
	i += 1

	#print(pred)
	

print(sum_time/i)

# Convert list to array and time on x-axis
np_prediction = np.asarray(prediction)
np_prediction = np.transpose(np_prediction)
	

plt.subplot(511)
plt.plot(np.transpose(input_data)[14])
plt.subplot(512)
plt.plot(np_prediction[0])
plt.subplot(513)
plt.plot(np_prediction[1])
plt.subplot(514)
plt.plot(np_prediction[2])
plt.subplot(515)
plt.plot(np_prediction[3])

plt.show()
