# Read in Relevant Packages
from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
import datetime
import time
from sklearn import metrics
from scipy.stats import gaussian_kde
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

start = datetime.datetime(2010, 1, 1)
mid = datetime.datetime(2013,1,1)
end = datetime.datetime(2013, 6, 1)

fq = data.DataReader('F', 'quandl', start, end)

fq = fq.iloc[::-1]



fq["ds"] = fq.index
fq["y"] = np.log(fq["AdjClose"]).diff()
fq["bam"] = np.log(fq["AdjClose"])

fq["Dates"] = pd.to_datetime(fq["ds"])
fq["Year"] = fq["Dates"].dt.year
fq["Mo"] = fq["Dates"].dt.month
fq["Day"] = fq["Dates"].dt.day

def setting_train_test(mid,end,column):
    X = (fq["Year"],fq["Mo"],fq["Day"])
    X = np.array(X).T
    Y = fq[column]
    x_whole = X[:end]
    y_whole = Y[:end]
    x_train = X[:mid]
    x_test = X[mid:end]
    y_train = Y[:mid]
    y_test = Y[mid:end]
    return mid,end,x_whole,x_train,x_test,y_whole,y_train,y_test

kernel = C(1, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=RBF(length_scale=20),alpha = .1,normalize_y=True,n_restarts_optimizer=15)

mid,end,x_whole,x_train,x_test,y_whole,y_train,y_test = setting_train_test(754,858,column = "bam")

gp.fit(x_train, y_train)
y_pred, sigma = gp.predict(x_whole, return_std=True)

plt.plot(fq.index,y_pred,color="G")
# plt.plot(test_bam["ds"],test_bam["bam"])
# plt.plot(train_bam["ds"],train_bam["y"])
plt.show()

print("The Root Mean Squared Error is: "+str(np.sqrt(metrics.mean_squared_error(y_whole,y_pred))))
print("The Mean Squared Error is: "+str(metrics.mean_squared_error(y_whole,y_pred)))
print("The MAPE is: "+str(mean_absolute_percentage_error(y_whole,y_pred)))
print("The Explained Variance Score is: "+str(metrics.explained_variance_score(y_whole,y_pred)))
print("The R Squared Score is: "+str(metrics.r2_score(y_whole,y_pred)))

