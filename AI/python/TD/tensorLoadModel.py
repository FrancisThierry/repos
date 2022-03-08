import os
from sklearn.feature_selection import SequentialFeatureSelector

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras


new_model = tf.keras.models.load_model('modelssaved/modelCar.h5')

new_model.compile(optimizer='adam',loss='mean_absolute_error')
# Check its architecture
new_model.summary()


new_gem = [[2020,10000]]

pred = new_model.predict(new_gem)

print("prediction",pred)