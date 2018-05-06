import numpy as np
import datetime
from pandas_datareader import data
import pandas as pd

start = datetime.datetime(2011, 1, 1)
end = datetime.datetime(2018, 1, 1)

fq = data.DataReader('LDOS', 'quandl', start, end)

fq = fq.iloc[::-1]

fq = fq.resample('M').mean()


fq.to_csv("/Users/williamnutter/Data_Science/R_Projects/LDOS_data.csv")