import csv
import numpy
import matplotlib.pyplot as plt
import pandas
import statsmodels.api as st
from statsmodels.tsa.arima_model import ARIMA
import time
import warnings
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn import linear_model
from sklearn import cross_validation

def dataProcessForARMA(filename):
    csvFile = open(filename, "r")
    reader = csv.reader(csvFile)

    data2 = []

    for item in reader:
        #print(item)
        #data = numpy.array(item)
        data = item

        for item2 in data:
            item2 = float(item2)
            data2.append(item2)
    csvFile.close()

    print(data2)
    data = data2
    serialNumberP = len(data2)
    print(serialNumberP)
    serialNumber = []
    
    for i in range(1, serialNumberP+1):
        #i = i + 1
        serialNumber.append(i)
    
    #print(len(data))
    #print(len(serialNumber))
    
    data4 = pandas.DataFrame({'Number': serialNumber, 'Energy':data})
    data2 = pandas.Series(data2)
    data3 = data2
    data2.index = pandas.Index(st.tsa.datetools.dates_from_range('1841','2248'))
    #print(data2)
    
    return(data, data2, data3, data4)

def dataProcessForLSTM(filename, seqlen, normalise_window):
    f = open(filename, 'r').read()
    data2 = f.split('\n')
    #print(data2)
    data = []
    for item2 in data2:
        #print(item2)
        f2 = item2.split(',')
        for item in f2:
            
            data.append(item)

    sequence_length = seqlen + 1
    #print(data)
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])  
        
    if normalise_window:
        result, win = normalise_windows(result)

    #print(result)
    result = numpy.array(result)
    #print(len(result))
    #print(len(win))

    row = round(0.8 * result.shape[0])
    train = result[:row, :]
    numpy.random.shuffle(train)
    xtrain = train[:, :-1]
    ytrain = train[:, -1]
    xtest = result[row:, :-1]
    ytest = result[row:, -1]
    print(ytest)
    print(len(ytest))
    win = win[row:row+72]
    
    xtrain = numpy.reshape(xtrain, (xtrain.shape[0], xtrain.shape[1], 1))
    xtest = numpy.reshape(xtest, (xtest.shape[0], xtest.shape[1], 1))


    return [xtrain, ytrain, xtest, ytest, win]

def autocorrelation(y, lags):
    n = len(y)
    #x = numpy.array(x)
    yMean = numpy.mean(y)
    denominator = 0
    for t in range(1, n):
        oneoftheDenominator = (y[t]-yMean)*(y[t]-yMean)
        denominator = denominator + oneoftheDenominator
    results = []
    
    for k in range(1, lags+1):
        molecule = 0
        for t in range(k+1, n):
            
            oneoftheMolecule = (y[t]- yMean)*(y[t-k]-yMean)
            molecule = molecule + oneoftheMolecule
            #oneoftheDenominator = 
        result = molecule/denominator
        
        results.append(result)

    return results

def accuracyMeasure(originalData, forecastData):
    x = originalData
    y = forecastData
    a = x - y
    a = abs(a)
    meanAbsoluteErrorresult = a.mean()
    b = a*a
    rmseResule = (b.mean())**(1/2)
    p = abs(100*a/x)
    mapeResult = p.mean()
    result = [meanAbsoluteErrorresult, rmseResule, mapeResult]
    print ('MAE is', meanAbsoluteErrorresult,', RMSE is', rmseResule, ', MAPE is', mapeResult)
    return result


def armaPredict(data2, data3):
    fig1 = plt.figure('data.png')
    plt.plot(data3)
    fig1.savefig('data.png')

    diff1 = data3.diff(1)
    diff2 = data3.diff(2)
    diff3 = data3.diff(5)

    fig2 = plt.figure('diff.png')

    plt.plot(diff3,label="diff3",color="red",linewidth=2)
    plt.plot(diff2,label="diff2",color='blue',linewidth=2)
    plt.plot(diff1,label="diff1",color="green",linewidth=2)
    plt.legend()
    fig2.savefig('diff.png')

    step = 50
    xlay = numpy.arange(step)+1   
    dataACF = autocorrelation(data3, step)
    print('AFC is', dataACF)
    fig3 = plt.figure('AFC.png')
    plt.bar(xlay, dataACF,width = 0.1, label="AFC",color="blue")
    plt.title("AFC") 
    plt.legend()
    fig3.savefig('AFC.png')

    fig4 = plt.figure('AFC2.png')
    fig4 = st.graphics.tsa.plot_pacf(data3,lags=50,ax = fig4.add_subplot(211))
    fig4.savefig('PAFC.png')

    arma = st.tsa.ARMA(data2,(24,3)).fit()
    print(arma)

    predict = arma.predict('2148', '2248', dynamic=True)
    print(predict)
    fig5, ax = plt.subplots()
    ax = data2.ix['1841':].plot(ax=ax)
    predict.plot(ax=ax)
    fig5.savefig('predict.png')
    
    return  predict

def build_model(layers):  
    model = Sequential()

    model.add(LSTM(input_dim=layers[0],output_dim=layers[1],return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(layers[2],return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

def normalise_windows(window_data):
    normalised_data = []
    win = []
    for window in window_data:   
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
        win.append(window[0])
    #print(win)
    return normalised_data, win

def lstmPredict(model, data):
    data = numpy.array(data)
    predicted = model.predict(data)
    print('predicted shape:',numpy.array(predicted).shape)  
    predicted = numpy.reshape(predicted, (predicted.size,))
    return predicted

def plotResults(predicted_data, true_data, filename):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.savefig(filename+'.png')

def typeTrans(data):
    data2 = []
    for item in data:
        item = float(item)
        data2.append(item)
    data2 = pandas.Series(data2)
    return(data2)   

filename = "2004.csv"
epochs  = 1
seq_len = 47

#data, data2, data3, data4 = dataProcessForARMA(filename)
#armaForecast = armaPredict(data2, data3)
#data6 = data2[307:408]
#plotResults(armaForecast, data6, 'ARMA Forecast')
#aMeasure = accuracyMeasure(data6, armaForecast)
#print('Accuracy measure for ARMA is: MAE:', aMeasure[0], ', RMSE:',aMeasure[1],", MAPE:", aMeasure[2] )

xTrain, yTrain, xTest, yTest, win = dataProcessForLSTM('2004.csv', seq_len, True)
model = build_model([1, 50, 100, 1])
model.fit(xTrain,yTrain,batch_size=512,nb_epoch=epochs,validation_split=0.05)
lstmForecast = lstmPredict(model, xTest)
yTest = typeTrans(yTest)
lstmForecast = typeTrans(lstmForecast)
win = typeTrans(win)
yTest = (yTest+1)*win
lstmForecast =(lstmForecast+1)*win
plotResults(lstmForecast,yTest,'LSTM Predictions')
aMeasure = accuracyMeasure(yTest,lstmForecast)
print('Accuracy measure for LSTM is: MAE:', aMeasure[0], ', RMSE:',aMeasure[1],", MAPE:", aMeasure[2] )

#X = numpy.array(data4[['Number']])
#Y = numpy.array(data4['Energy'])
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
#clf = linear_model.LinearRegression()
#clf.fit (X_train,y_train)
#a = clf.predict(X_test)
#plotResults(a,y_test,'123')
#s = accuracyMeasure(y_test, a)
#print('Accuracy measure for Linear Regression is: MAE:', aMeasure[0], ', RMSE:',aMeasure[1],", MAPE:", aMeasure[2] )
