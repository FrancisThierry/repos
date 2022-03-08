#Imports
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
#Scale data to be between 0 and 1
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
cols = ['odometer','price','year','manufacturer','model_name']
df = pd.read_csv('data/carsTDIA.csv',usecols=cols)
# df.query('manufacturer=="Ford"',inplace=True)
df.drop(columns=['manufacturer','model_name'],inplace=True)

df.head()

print('df',df)



#We need .values because it's best to pass in numpy arrays due to how tensorflow works
X = df[['year', 'odometer']].values
y = df['price'].values
#Split into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


scaler = MinMaxScaler()

#Calc params needed to scale later on
#Only perform on training set as to not influence based on test data
scaler.fit(X_train)

#Perform transformation
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Add multiple layers into sequential with the number of neurons needed
model = Sequential()

model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(4,activation='relu'))

model.add(Dense(1))

model.compile(optimizer='adam',loss='mean_absolute_error')

#Fit
model.fit(x=X_train,y=y_train,epochs=100)

#Grab losses and stick into a dataframe
loss_df = pd.DataFrame(model.history.history)



#Evaluate MSE for test vs training set
training_score = model.evaluate(X_train,y_train,verbose=0)
test_score = model.evaluate(X_test,y_test,verbose=0)

print("training score", training_score)
print("training score", test_score)

test_predictions = model.predict(X_test)

new_gem = [[1998,200000]]

#Remember to scale the data
new_gem = scaler.transform(new_gem)

pred = model.predict(new_gem)

print("prediction",pred)

# Save the entire model as a SavedModel

model.save('modelssaved/modelCar.h5')

