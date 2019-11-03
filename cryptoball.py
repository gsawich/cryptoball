from keras import layers
from keras.models import Sequential
from time import strftime, gmtime
import urllib
import json
import numpy
import matplotlib
import matplotlib.pyplot as plotter
import time

def normalize(data):
    result = []
    for window in data:
        winArray = []
        for p in window:
            calc = (float(p)/float(window[0])) - 1
            winArray.append(calc)
        result.append(winArray)
    return result

def predict_sequences_multiple(model, data, window_size, prediction_len):
    #Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data)/(prediction_len))+1):
        curr_frame = data[i*prediction_len]
        predicted = []
        for j in range(prediction_len):
            predicted.append(model.predict(curr_frame[numpy.newaxis,:,:])[0,0])
            curr_frame = curr_frame[1:]
            curr_frame = numpy.insert(curr_frame, [window_size-1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs

BINS = 49
EP = 1
WINDOWS = 1
PAIR = 'BTC_ETH'
candlestick = 1800
timespan = 21600*4*180
lookahead = 6

def predict():

    currentTime = int(time.time());

    data = urllib.request.urlopen("https://poloniex.com/public?"
                                  "command=returnChartData&currencyPair=" + PAIR +
                                  "&start=" + str(currentTime-timespan)  + "&end=" + str(currentTime) + "&period="+str(candlestick))
    string = data.read().decode('utf-8')
    obj = json.loads(string)
    x_in = list()
    y_in = list()
    z_in = list()
    w_in = list()
    w1 = list()
    w2 = list()
    for day in obj:
        for attr, v in day.items():
            if attr == 'date':
                x_in.append(v)
            if attr == 'weightedAverage':
                y_in.append(v)
            if attr == 'volume':
                z_in.append(v+0.01)
            if attr == 'high':
                w1.append(v)
            if attr == 'low':
                w2.append(v)

    price = []
    vol = []
    deviation = []
    for k,m in enumerate(w1):
        w_in.append(abs(m-w2[k])+0.001)

    for i in range(len(x_in) - lookahead):
        price.append(y_in[i: i + lookahead])
        vol.append(z_in[i: i + lookahead])
        deviation.append(w_in[i: i + lookahead])

    price = normalize(price)
    deviation = normalize(deviation)
    maxim = numpy.amax(price)
    maxvol = numpy.amax(vol)
    maxdev = numpy.amax(deviation)

    for k,m in enumerate(vol):
        for l,n in enumerate(m):
            vol[k][l] = (n/maxvol)*maxim
            deviation[k][l] = (deviation[k][l] / maxdev)*maxim

    bins = numpy.linspace(start=-maxim, stop=maxim, num=BINS)
    #volbins = numpy.logspace(start=-maxim, stop=maxim, num=BINS)
    #print(vol)
    discprice = list()
    for j,win in enumerate(price):
        volWin = vol[j]
        devWin = deviation[j]
        newWin = list()
        hist = numpy.digitize(win, bins)
        #volhist = numpy.digitize(volWin, volbins)
        #print(volhist)
        for i,x in enumerate(hist):
            bit = list()
            bit.append(bins[x-1])
            bit.append(volWin[i])
            bit.append(devWin[i])
            newWin.append(bit)
            discprice.append(newWin)
    price = discprice
    #vol = normalize(vol)
    price = numpy.array(price)
    #rint(price)
    #vol = numpy.array(vol)
    row = round(0.9 * price.shape[0])
    train = price[:int(row), :]
    #numpy.random.shuffle(train)
    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = price[int(row):, :-1]
    y_test = price[int(row):, -1]

    #print(x_train.shape)
    #print(y_train.shape)

    x_train = numpy.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))
    x_test = numpy.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

    #y_train = numpy.reshape(y_train, (y_train.shape[0], 1))
    #y_test = numpy.reshape(y_test, (y_test.shape[0],  1))

    #print(x_test.shape)

    model = Sequential()

    model.add(layers.recurrent.LSTM(
        units=(lookahead*2),
        input_shape=(lookahead-1, 3),
        return_sequences=True
    ))
    model.add(layers.core.Dropout(0.2))

    model.add(layers.recurrent.LSTM(
        (lookahead*4),
        return_sequences=False
    ))
    model.add(layers.core.Dropout(0.2))

    model.add(layers.core.Dense(
        3))
    model.add(layers.core.Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop', metrics=['accuracy'])

    #transition = numpy.concatenate((x_in, y_in), axis=0)
    #input = numpy.concatenate((transition, z_in), axis=0)

    model.fit(
        x_train,
        y_train,
        batch_size = 1024,
        epochs=EP,
        validation_split=0.45,
        verbose=1
    )

    x_pred = []
    for row in x_test:
        #print(row)
        x_temp = []
        for x in row:
            x_temp.append(x[0])
        x_pred.append(x_temp)
    x_pred = numpy.array(x_pred)
    x_pred = numpy.reshape(x_pred, (x_pred.shape[0], x_pred.shape[1], 1))
    #print(x_pred)

    predictions = predict_sequences_multiple(model, x_test, (lookahead-1), lookahead*WINDOWS)

    fig = plotter.figure(facecolor = 'white')
    ax = fig.add_subplot(111)
    ax.plot(y_test, label='True')
    fortune = 0
    currPrice = y_in[len(y_in)-1]
    #ay = fig.add_subplot(111)
    #ay.plot(y_in, label='Value')
    for i, d in enumerate(predictions):
        padding = list()
        for j,v in enumerate(d):
            d[j] = v
            padding.append(j*2 + (i*lookahead*WINDOWS))
            if (i == (len(predictions)-1) and j == (len(d)-1)) :
                fortune += (v * currPrice)
        plotter.plot(padding, d, label = 'Prediction')

        #plotter.legend()

    ann = "Resolution: " + str(candlestick/60) + "mins \n Lookahead: " \
          + str((candlestick*lookahead)/3600) + "hrs \n Timeframe: " + str((timespan/3600)/24) \
          + "days \n Predicted Price: " + str(fortune+currPrice) + " \n Current Price: " + str(currPrice)
    ax.text(0,0.1, ann)

    predPrice = (fortune+currPrice)
    predT = (currentTime + (candlestick)) - (6*3600)
    predTime = strftime('%d-%m-%y  %H:%M:%S', time.gmtime(predT))

    print("Predicted Price: " + str(predPrice) + " \n Current Price: " + str(currPrice) + "\n At: " + predTime)

    #fig2 = plotter.figure(facecolor = 'white')
    #ay = fig2.add_subplot(111)
    #ay.plot(y_in)
    #ay.text(0,0, "Current Price: " +str(currPrice))
    plotter.show()
    return (str(predTime) + " = " + str(predPrice))

startTime =  int(time.time())
hasP = False

while(True) :
    t = int(time.time())
    #print((t - startTime) % 60)

    if (((t - startTime) % 60 < 5) & (hasP == False)):
        hasP == True
        pred = predict()

        fname = PAIR + '.txt'
        f = open(fname, 'a+')

        f.write(pred)
        f.write("\n")
        f.close()
        hasP = False