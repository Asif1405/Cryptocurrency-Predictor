SEQ_LEN = 60  # how long of a preceeding sequence to collect for RNN
FUTURE_PERIOD_PREDICT = 3 
RATIO_TO_PREDICT = "LTC-USD"
EPOCHS = 10 
BATCH_SIZE = 64 
NAME = f"{SEQ_LEN}-SEQ-{FUTURE_PERIOD_PREDICT}-PRED-{int(time.time())}"


def classify(current, future):
    if float(future) > float(current):  # if the future price is higher than the current then 1
        return 1
    else: 
        return 0


def preprocess_df(df):
    df = df.drop("future", 1)

    for col in df.columns: 
        if col != "target":  # normalize data, except for the target
            df[col] = df[col].pct_change() 
            df.dropna(inplace=True) 
            df[col] = preprocessing.scale(df[col].values)  # scaling data between 0 and 1.

    df.dropna(inplace=True)

    #sequencing data
    sequential_data = [] 
    prev_days = deque(maxlen=SEQ_LEN)
    for i in df.values:  
        prev_days.append([n for n in i[:-1]]) 
        if len(prev_days) == SEQ_LEN: 
            sequential_data.append([np.array(prev_days), i[-1]]) 

    random.shuffle(sequential_data) 

    buys = []  
    sells = []  

    for seq, target in sequential_data:
        if target == 0:  
            sells.append([seq, target])  
        elif target == 1:  
            buys.append([seq, target]) 

    random.shuffle(buys)  
    random.shuffle(sells)  

    lower = min(len(buys), len(sells))  

    # make sure both lists are only up to the shortest length.
    buys = buys[:lower] 
    sells = sells[:lower]  

    sequential_data = buys+sells  
    random.shuffle(sequential_data)  

    X = []
    y = []

    for seq, target in sequential_data: 
        X.append(seq)  
        y.append(target) 

    return np.array(X), y  

#creating main dataframe
main_df = pd.DataFrame() 

ratios = ["BTC-USD", "LTC-USD", "BCH-USD", "ETH-USD"] 
for ratio in ratios: 

    ratio = ratio.split('.csv')[0]  
    print(ratio)
    dataset = f'crypto_data/{ratio}.csv'  
    df = pd.read_csv(dataset, names=['time', 'low', 'high', 'open', 'close', 'volume'])  

    df.rename(columns={"close": f"{ratio}_close", "volume": f"{ratio}_volume"}, inplace=True)

    df.set_index("time", inplace=True) 
    df = df[[f"{ratio}_close", f"{ratio}_volume"]] 

    if len(main_df)==0:  
        main_df = df  
    else: 
        main_df = main_df.join(df)

main_df.fillna(method="ffill", inplace=True) 
main_df.dropna(inplace=True)

main_df['future'] = main_df[f'{RATIO_TO_PREDICT}_close'].shift(-FUTURE_PERIOD_PREDICT)
main_df['target'] = list(map(classify, main_df[f'{RATIO_TO_PREDICT}_close'], main_df['future']))

main_df.dropna(inplace=True)

times = sorted(main_df.index.values)
last = sorted(main_df.index.values)[-int(0.05*len(times))]

#creating train and test sets
test_df = main_df[(main_df.index >= last)]
main_df = main_df[(main_df.index < last)]

#dividing train and test sets
train_x, train_y = preprocess_df(main_df)
test_x, test_y = preprocess_df(test_df)

print(f"train data: {len(train_x)} validation: {len(test_x)}")
print(f"Dont buys: {train_y.count(0)}, buys: {train_y.count(1)}")
print(f"VALIDATION Dont buys: {test_y.count(0)}, buys: {test_y.count(1)}")

#Building Model
model = Sequential()
model.add(CuDNNLSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128, return_sequences=True))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(2, activation='softmax'))

#defining optimizer
optimizer = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compiling model
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Training model
history = model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(validation_x, validation_y))

# Score model
score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
# Save model
model.save("models/{}".format(NAME))
