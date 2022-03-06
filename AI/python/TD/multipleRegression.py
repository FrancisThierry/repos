from sklearn import linear_model
import pandas as pd
cars = pd.read_csv('data/sample.csv')

X = cars[['year', 'odometer']]
y = cars['price']


regr = linear_model.LinearRegression()
regr.fit(X, y)

predictPrice = regr.predict([[2022, 150000]])
print(predictPrice)