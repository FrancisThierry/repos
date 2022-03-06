import pandas as pd
import numpy as np

# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

CAR_TEST = "data/test.csv"
CAR_TRAINING = "data/train.csv"


car_trainig = pd.read_csv(CAR_TRAINING)

car_trainig.head()

car_data_test = pd.read_csv(CAR_TEST)
car_label = car_data_test.pop('price')

car_data_test = np.array(car_data_test)


model = tf.keras.Sequential([
  layers.Dense(64),
  layers.Dense(1)
])

model.compile(loss = tf.losses.MeanSquaredError(),
                      optimizer = tf.optimizers.Adam())

model.fit(car_data_test, car_label, epochs=40)                      

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(car_data_test, car_label, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions")
topred = [2022]
predictions = model.predict(car_data_test)
print("predictions shape:", predictions.shape)
print("predictions :", predictions)