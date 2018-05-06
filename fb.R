library(forecast)

x <- function() {
start_time <- Sys.time()
data <- read.csv(file = "LDOS_data.csv")
t <- ts(data$AdjOpen[1:(6*13)],frequency=6)
print(length(data$AdjOpen))
x <- msts(data$AdjOpen, seasonal.periods=c(2,3))
auto = auto.arima(t,D=1)

f_30 = forecast(auto,h=6,level=.5)
f_100 = forecast(auto,h=10)
f_700 = forecast(auto,h=100)
ts.plot(f_30[4]$mean,ts(data$AdjOpen)[((6*13)+1):84],gpars = list(col = c("black", "red")),main="test")
plot(f_30)
plot(f_100)
plot(f_700)
plot(ts(data$AdjOpen))

end_time <- Sys.time()

print(end_time - start_time)
y1 = data$AdjOpen[6*13]
arr_30 = sapply(f_30[4],function(x) (x-y1)/y1)
arr_100 = sapply(f_100[4],function(x) (x-y1)/y1)
arr_700 = sapply(f_700[4],function(x) (x-y1)/y1)

plot(arr_30)
plot(arr_100)
plot(arr_700)
}

ROR <- function(x,y_1 = 0) {
  return (x-y_1)/y_1
}