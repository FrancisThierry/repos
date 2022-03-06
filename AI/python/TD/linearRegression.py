from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
cars = pd.read_csv('data/sample.csv')


x = cars['year']
y = cars['price']


slope, intercept, r, p, std_err = stats.linregress(x, y)

def myLinearRegressionFunction(x):
  return slope * x + intercept

mymodel = list(map(myLinearRegressionFunction, x))


price = myLinearRegressionFunction(2022)

print(price)

plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()
