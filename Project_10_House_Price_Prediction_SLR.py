# In these we use simple linear regression model to predict housing prices.

#Import libraries 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
# np.set_printoptions(threshold=np.nan)

#Import Dataset

dataset = pd.read_csv(r"C:\Users\darsh\OneDrive\Desktop\FSDS\FSDS_21_03\20th- slr\SLR - House price prediction\House_data.csv")
space = dataset['sqft_living']
price = dataset['price']

x= np.array(space).reshape(-1, 1)
y = np.array(price)

#Splitting the data into Train & Test

from sklearn.model_selection  import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = 1/3,random_state = 0)

#Fitting simple linear regression to the Training Set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

#Predicting the price

pred = regressor.predict(xtest)

#Visualizing the training Test Results

plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color = 'blue')
plt.title("Visuals for Training Dataset")
plt.xlabel("Space")  
plt.ylabel("Price")
plt.show()

#Visualizing the Test Results

plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color ='blue')
plt.title("Visuals for Test dataset")
plt.xlabel("Space")
plt.ylabel("Price")
plt.show()








