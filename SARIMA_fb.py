# Read in Relevant Packages
from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
import datetime
import pandas as pd
from sklearn import metrics
import statsmodels.tsa.seasonal as ssnl
from statsmodels.graphics import tsaplots
from pandas import Series as Series
from statsmodels.tsa.stattools import adfuller,arma_order_select_ic,acf,pacf
from statsmodels.tsa.statespace import sarimax
from statsmodels.tsa.x13 import x13_arima_select_order as x13

# create a differenced series

def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)

# invert differenced forecast
def inverse_difference(last_ob, value):
    return value + last_ob

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def arma_test(ts,r = 5):

    best_aic = np.inf
    best_order = None
    best_mdl = None

    rng = range(r)
    for i in rng:
        for j in rng:
            try:
                tmp_mdl = smt.ARMA(ts.values, order=(i, j)).fit(method='mle', trend='nc')
                print("("+str(i)+","+str(j)+") AIC: "+str(tmp_mdl.aic))
                tmp_aic = tmp_mdl.aic
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (i, j)
                    best_mdl = tmp_mdl
            except: continue


    print('aic: {:6.5f} | order: {}'.format(best_aic, best_order))


# Create the relevant time series dataframes

start = datetime.datetime(2010, 1, 1)
mid = datetime.datetime(2013,1,1)
end = datetime.datetime(2013, 6, 1)

fq = data.DataReader('F', 'quandl', start, end)

fq = fq.iloc[::-1]

fq["ds"] = fq.index
fq["y"] = np.log(fq["AdjClose"]).diff()
fq["bam"] = np.log(fq["AdjClose"])
fq["firstdiff"] = fq.y -fq.y.shift(1)
train = fq.loc[start:mid,["ds","y"]].dropna(inplace=False)
train_bam = fq.loc[start:mid,["ds","bam"]].dropna(inplace=False)
test = fq.loc[mid:end,["ds","y"]].dropna(inplace=False)
test_bam = fq.loc[mid:end,["ds","bam"]].dropna(inplace=False)
#

def plt_diff():
    plt.plot(fq["ds"],fq["y"])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
    return None

def plt_decomposition():
    decomposition = ssnl.seasonal_decompose(train["y"],freq=240)
    fig = decomposition.plot()
    fig.set_size_inches(15, 8)
    plt.show()
    return None

def test_stationarity(timeseries):
    # Determing rolling statistics
    rolmean = pd.rolling_mean(timeseries, window=12)
    rolstd = pd.rolling_std(timeseries, window=12)

    # Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()

    # Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

#test_stationarity(train["y"])

def plt_acf_pacf():
    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(211)
    tsaplots.plot_acf(train["y"],lags=240,ax=ax1)
    ax2 = fig.add_subplot(212)
    tsaplots.plot_pacf(train["y"],lags=240,ax=ax2)
    plt.show()
    return None

# res = arma_order_select_ic(train["y"],max_ar=4,max_ma=4,ic=["aic","bic"])
# print(res.aic_min_order)
# print(res.bic_min_order)
#print(res)

def get_pa_indecies():
    a = acf(train["y"],nlags = 240)
    p = pacf(train["y"],nlags = 240)
    ab = a[np.abs(a)>(1.96/np.sqrt(len(train["y"])))]
    pb = p[np.abs(p)>(1.96/np.sqrt(len(train["y"])))]
    print("acf")
    print(np.where(np.isin(a,ab)))
    print("pacf")
    print(np.where(np.isin(p, pb)))
    return None

#get_pa_indecies()

mod = sarimax.SARIMAX(train_bam["bam"].values, trend='n', order=(2,0,1), seasonal_order=(2,1,0,180))
results = mod.fit()
#print(results.summary())

test_bam['forecast'] = results.forecast(104)
fig = plt.figure(figsize=(12,8))
print("train_bam: "+str(test_bam["forecast"].shape))
print("test_bam: "+str(test_bam["bam"].shape))
plt.plot(train_bam["ds"],train_bam['bam'])
plt.plot(test_bam["ds"],test_bam["forecast"])
plt.show()


print("The Root Mean Squared Error is: "+str(np.sqrt(metrics.mean_squared_error(test_bam["bam"],test_bam["forecast"]))))
print("The Mean Squared Error is: "+str(metrics.mean_squared_error(test_bam["bam"],test_bam["forecast"])))
print("The MAPE is: "+str(mean_absolute_percentage_error(test_bam["bam"],test_bam["forecast"])))
print("The Explained Variance Score is: "+str(metrics.explained_variance_score(test_bam["bam"],test_bam["forecast"])))
print("The R Squared Score is: "+str(metrics.r2_score(test_bam["bam"],test_bam["forecast"])))
plt.plot(test_bam["ds"],test_bam["bam"])
plt.plot(test_bam["ds"],test_bam["forecast"])
plt.savefig("Sarima.png")
plt.show()
plt.scatter(test_bam["bam"],test_bam["forecast"])
plt.show()