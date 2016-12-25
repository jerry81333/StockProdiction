from yahoo_finance import Share
from pandas import Series,DataFrame
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
import statsmodels.tsa.stattools as ts
from scipy import  stats
import pywt

yaho = Share('YHOO') #choose stock, YAHOO, GOLD
startday='2015-11-1' #choose first day
endday='2016-12-15' #choose end day
#train = 15 #How many data for train, 9 is the least.

#draw
fig=plt.figure()
ax1=fig.add_subplot(711)
ax2=fig.add_subplot(712)
ax3=fig.add_subplot(713)
ax4=fig.add_subplot(714)
ax5=fig.add_subplot(715)
ax6=fig.add_subplot(716)
ax7=fig.add_subplot(717)

#Data processing
StockDate = DataFrame(yaho.get_historical(startday, endday))
StockDate.index = StockDate.Date
StockDate = DataFrame.sort_index(StockDate) #sort

test = DataFrame(yaho.get_historical(startday, '2016-12-1'))
test.index = test.Date
test = DataFrame.sort_index(test)
test = test['Close']
test=test.astype(float)
test.plot(ax=ax5)

#L=len(StockDate)
#total_predict_data=L-train

'''
#draw
Data = StockDate.drop(['Date','Symbol','Adj_Close'],axis=1) 
Data=Data.astype(float)
ax=Data.plot(secondary_y=['Volume'])
ax.set_ylabel('Value')
ax.right_ax.set_ylabel('Volume')
plt.grid(True)
plt.show()
#Create more data
value = pd.Series(Data['Open']-Data['Close'],index=Data.index)
#Data['DOP'] = value #Difference between Open and Close
#Data['DHL'] = Data['High']-Data['Low'] #Difference between High and Low
value[value>=0]=0 #0 means fall
value[value<0]=1 #1 means rise
print(value)
'''
#ARIMA
Close_original = StockDate['Close']
Close_original=Close_original.astype(float)
#Close.plot()
close=pywt.dwt(Close_original, 'db4') #DB4,Wavelet decomposition
Close_db4=pd.Series(close[0])
Close_db4=Close_db4-14
Close_db4.index = pd.Index(sm.tsa.datetools.dates_from_range('2001','2145'))
#aa=Close.diff(3)

#draw
#aa.plot(ax=ax4)
Close_db4.plot(ax=ax2)
Close=Close_db4.diff(4) #stationary time series
Close=Close[4:]

print("Augmented Dickey-Fuller test:",ts.adfuller(Close,4)) #Augmented Dickey-Fuller test
Close.plot(ax=ax3)
Close_original.plot(ax=ax1)

sm.graphics.tsa.plot_acf(Close,lags=40,ax=ax6) #ARIMA,q
sm.graphics.tsa.plot_pacf(Close,lags=40,ax=ax7) #ARIMA,p
#print(Close)

Arma = sm.tsa.ARMA(Close,order=(9,3)).fit(disp=-1, method='mle')
print(Arma.aic,Arma.bic,Arma.hqic)

#predict
Arma_stock=Arma.predict()
Arma_stock.plot(ax=ax3)
predict_stock = Arma.predict('2137','2148',dynamic=True)
predict_stock.plot(ax=ax3)

#reduce diff()
L=len(Arma_stock)
i=0
while i<L:
	if(i<4):
		Arma_stock[i]=Arma_stock[i]+Close_db4[i]
	else:
		Arma_stock[i] = Arma_stock[i]+Arma_stock[i-4]
	i=i+1
Arma_stock.plot(ax=ax4)
L=len(predict_stock)
i=0
while i<L:
	if(i<4):
		predict_stock[i] = predict_stock[i]+Arma_stock[-4+i]
	else:
		predict_stock[i] = predict_stock[i]+predict_stock[i-4]
	i=i+1	
predict_stock.plot(ax=ax4)

plt.grid(True)
plt.show()

'''
#Data['Value']=value
#SVM
correct = 0
train_original=train
while train<L:
	Data_train=Data[train-train_original:train]
	value_train = value[train-train_original:train]
	Data_predict=Data[train:train+1]
	value_real = value[train:train+1]
	#print(Data_train)
	#print(value_train)
	print(train)
	#classifier =svm.SVC(kernel='poly') #52% need optimization, some data may expand to infinite demension
	#classifier =svm.SVC(kernel='sigmoid') #49%
	#classifier =svm.SVC(kernel='precomputed') #bug
	#classifier =svm.SVC() #kernel='rbf'
	classifier =svm.LinearSVC() #53%
	classifier.fit(Data_train,value_train)
	print(train)
	value_predict=classifier.predict(Data_predict)
	print(train)
	#print("value_real = ",value_real[0])
	#print("value_predict = ",value_predict)
	if(value_real[0]==int(value_predict)):
		correct=correct+1
		print(correct)
	train = train+1
correct=correct/total_predict_data*100
print("Correct =",correct,"%")

#test SVM
print("support_:",classifier.support_)
print("support_vectors_:",classifier.support_vectors_)
print("n_support_:",classifier.n_support_)
'''

