import pandas
from pandas_datareader import data
import matplotlib.pyplot as plt
import numpy as np
import fbprophet
import datetime
from sklearn import metrics

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
start = datetime.datetime(2010, 1, 1)
mid = datetime.datetime(2013,1,1)
end = datetime.datetime(2013, 6, 1)

# fm = data.DataReader('F', 'morningstar', start, end)

fq = data.DataReader('F', 'quandl', start, end)

fq = fq.iloc[::-1]


# plt.tight_layout()

# fq["logAdjClose"] = fq.loc[:,"AdjClose"].apply(np.log)
# fq["diffAdjClose"] = fq["logAdjClose"].diff()
# fq["diffAC"] = fq["AdjClose"].diff()
# plt.plot(fq.loc[mid:end]["diffAC"])
# plt.plot(fq.loc[mid:end]["diffAdjClose"])
# plt.show()
# fig = plt.figure(1,figsize=[10,6])
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)
# ax1.set_title("Adjusted Close")
# ax2.set_title("Log Adjusted Close")
# ax3.set_title("Differenced Log AdjClose")
# ax1.plot(fq.index, fq["AdjClose"])
# ax2.plot(fq.index, fq["logAdjClose"])
# ax3.plot(fq.index, fq['diffAdjClose'])
# plt.tight_layout()
# plt.show()
#print(fq.loc[start:mid]["diffAdjClose"].head())
fq["ds"] = fq.index
fq["y"] = np.log(fq["AdjClose"])
df = fq.loc[start:mid,["ds","y"]]
test = fq.loc[mid:end,["ds","y"]]

gm_prophet = fbprophet.Prophet(changepoint_prior_scale=0.15)
gm_prophet.fit(df)

gm_forecast = gm_prophet.make_future_dataframe(periods=len(test), freq='D')


gm_forecast = gm_prophet.predict(gm_forecast)
yhat = gm_forecast[["yhat"]]
yhat = yhat.iloc[-len(test):]
print("The Root Mean Squared Error is: "+str(np.sqrt(metrics.mean_squared_error(test[["y"]],yhat))))
print("The Mean Squared Error is: "+str(metrics.mean_squared_error(test[["y"]],yhat)))
print("The MAPE is: "+str(mean_absolute_percentage_error(test[["y"]],yhat)))
print("The Explained Variance Score is: "+str(metrics.explained_variance_score(test[["y"]],yhat)))
print("The R Squared Score is: "+str(metrics.r2_score(test[["y"]],yhat)))
plt.plot(test["ds"],test["y"])
plt.plot(test["ds"],yhat)
plt.show()
plt.scatter(test[["y"]],yhat)
plt.show()
gm_prophet.plot(gm_forecast)
plt.show()
gm_prophet.plot_components(gm_forecast)
plt.show()

# for SARIMA you need to stationarize the data... lets do it

