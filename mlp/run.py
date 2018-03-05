import socket
import sys
import csv
import numpy
import os
import glob
import random
import numpy as np
import signal
import string
import sys

print(sys.argv)

from keras.models import Sequential
from keras.models import model_from_json

# Datalogger to save information
input_log = []
prediction_log = []

# Set udp communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configure UDP
receive_address = ('127.0.0.1', 5000)
send_address = ('127.0.0.1', 5001 )

print('Receiving on ' + str(receive_address) + ' port.')
print('Sending on ' + str(send_address) + ' port.\n')

# Open socket
sock.bind(receive_address)

# Define a function to close the socket, because if not the program block on recvfrom
def sigint_handler(signum, frame):

    # Save log files
    np.savetxt('input_log.csv', input_log, fmt='%.2f', delimiter=';')
    np.savetxt('prediction_log.csv', prediction_log, fmt='%.2f', delimiter=';')

 #   print(prediction_log)

    # Need to press twice CTRL-C
    #print("Press CTRL-C another time!")
    # Close socket
  #  sock.close()
    sys.exit(0)

# Sign the sigint_handler to CTRL-C and exit
signal.signal(signal.SIGINT, sigint_handler)

# Load json and create model
json_file = open('./model/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# Load weights int the new model
loaded_model.load_weights("./model/model.h5")
loaded_model.summary()

print("\nEnable self-driving mode on CONE-SIM...")

# Andreas Mikkelsen's Loop
while True:

    # Exception to socket
    try:
        # Receive data from the game in CSV format with ';'
        received_data, address = sock.recvfrom(4096)

        if received_data:

            # Split received data in a numpy array
            #telemetry = np.array(list(csv.reader(received_data, delimiter=";", quoting=csv.QUOTE_NONNUMERIC)))
            #telemetry = csv.reader(received_data, delimiter=';', quoting=csv.QUOTE_NONNUMERIC)
            telemetry = np.array(string.split(received_data, ';'), dtype=float)

            # Log received data
            #input_log.append(telemetry)
            #print(telemetry)

            # Remove some features to the format of the NN
            #data_p1 = telemetry[1:6]              # Throttle, brake, steering, handbrake, speed
            data_p1 = telemetry[5]
            data_p2 = telemetry[10:46]         # lidar from 0 to 180 degres

            # Append the input features (23 features)
            input_data = np.append(data_p1, data_p2)
            #input_data[4] /= 150.
            #input_data[5:] /= 15.

            # Cheat
            #input_data[:3] = 0

            #print(input_data)

            # Predict commands Throttle, brake, Steering, handbrake
            prediction = loaded_model.predict(input_data.reshape(1,37))
            prediction = prediction[0]  # [[]] -> [] I dont know how to explain... first row of 1 row matrix kkk

            #print(prediction)

            # Log predictions
            prediction_log.append(prediction)

            # Create a package of the commands to sent to the game
            #cmd_msg =   str(prediction[0]) + ";" + str(prediction[1]) + ";" + str(prediction[2]) + ";" + str(prediction[3])
            cmd_msg = '{0:.3f};{1:.3f};{2:.3f};{3:.3f}'.format(abs(prediction[0]), 0, prediction[2], 0)

            print(cmd_msg)

            sock.sendto(cmd_msg, send_address)
                

    finally:
        #print(received_data)
        #print("\n")
        #print("ops")
        pass
