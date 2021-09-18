from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def home(request):
    return render(request,'home.html')

def predict(request):
    return render(request,'predict.html')

def result(request):
    df = pd.read_csv('C:/Users/pc lenovo/Downloads/USA_Housing.csv')
    df.drop('Address', inplace=True, axis=1)
    X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
    y = df['Price']
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.30)
    model=LinearRegression()
    model.fit(X_train,y_train)
    var1 = float(request.GET['n1'])
    var2 = float(request.GET['n2'])
    var3 = float(request.GET['n3'])
    var4 = float(request.GET['n4'])
    var5 = float(request.GET['n5'])
    pred = model.predict(np.array([var1,var2,var3,var4,var5]).reshape(1,-1))
    pred = round(pred[0])
    price = "the predicted price is $"+str(pred)
    
    return render(request,'predict.html', {"result2":price})
