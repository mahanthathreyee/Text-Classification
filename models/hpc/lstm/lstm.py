import sys, os, re, csv, codecs, numpy as np, pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y_tr = train[list_classes].values
y_te = test[list_classes].values

list_sentences_train = train["comment_text"]
list_sentences_test = test["comment_text"]

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)

print('Fitting Tokenizer to Test Data')
tokenizer.fit_on_texts(list(list_sentences_train))

print('Tokenizing Train Data')
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

print('Tokenizing Test Data')
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

maxlen = 200
X_tr = pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = pad_sequences(list_tokenized_test, maxlen=maxlen)

inp = Input(shape=(maxlen, )) #maxlen=200 as defined earlier

embed_size = 128
x = Embedding(max_features, embed_size)(inp)

x = LSTM(60, return_sequences=True, name='lstm_layer')(x)

x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(50, activation="relu")(x)
x = Dropout(0.1)(x)
x = Dense(6, activation="sigmoid")(x)

print('Compiling Model')
model = Model(inputs=inp, outputs=x)
model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

batch_size = 32
epochs = 5
print('Training Model')
model.fit(
    X_tr, y_tr,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
    workers=4,
    use_multiprocessing=True,
    verbose=2
)

print('Training Summary')
print(model.summary())

print('Evaluating with Test Data')
print(model.evaluate(X_te, y_te))