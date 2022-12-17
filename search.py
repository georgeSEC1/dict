#BCI - brain computer interface technology. 
#Copyright (C) 2022 George Wagenkencht
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import time
import random
import string
partition = 10
sampleSize = 150000
defaultLength = 8
baud = 9600 
option = ""
dictumSize = 1
def delay(ngram):
    print()
    print(ngram)
    print(3)
    time.sleep(1)
    print(2)
    time.sleep(1)
    print(1)
    time.sleep(1)
    return
def resetDataFile(dataFile):
    f = open(dataFile, "w", encoding="utf8")
    f.close()
def returnNgrams(data,length, mode):
    if mode == "sequential":
        ngram = ""
        pos = random.randint(1,len(data))
        n = 0
        while(n < length and pos+length < len(data)-1):
            if pos+n < len(data)-2 and pos+n > 0:
                ngram += data[pos+n] + " "
            n+=1
        return ngram
    if mode == "random":
        ngram = ""
        pos = random.randint(1,len(data))
        n = 0
        while(n < length and pos+length < len(data)-1):
            pos = random.randint(1,len(data))
            if pos+n < len(data)-2 and pos+n > 0:
                ngram += data[pos+n] + " "
            n+=1
        return ngram
def predict(user,inputFile,model):#refactor into construction using gen() input rather than record() input
    db = []
    model = keras.models.load_model(model)
    with open(inputFile, encoding='ISO-8859-1') as f:
        textC = f.readlines()
    varX = textC[0].count(",")
    dataset = np.loadtxt(inputFile, delimiter=',',usecols = range(varX))
    X = dataset[:,:]
    predictions = (model.predict(X)).astype(int)
    for i in range(len(predictions)):
        if predictions[i] == 0:
            db.append(str(0))
        if predictions[i] == 1:
            delay(user)
            f = open("NewWords", "a", encoding="utf8")
            f.write(user)
            f.write("\n")
            f.close()
            db.append(str(1))
        print('%s => %d' % (X[i].tolist(), predictions[i]))
    return db
def train(dataFile,modelName):
    with open(dataFile, encoding='ISO-8859-1') as f:
        text = f.readlines()
    varX = text[0].count(",")
    dataset = np.loadtxt(dataFile, delimiter=',',usecols = range(varX+1))
    X = dataset[:,0:varX]
    y = dataset[:,varX]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
    model = Sequential()
    model.add(Dense(120, input_shape=(X_train.shape[-1],), activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, batch_size=10, verbose=1)
    _, accuracy = model.evaluate(X_test, y_test)
    print('Accuracy: %.2f' % (accuracy * 100))
    model.save(modelName)
def recordData(ngram,stress,dataFile):#Adversarial training between easy and difficult n-grams, full 2d grapheme differentiation...
    record = np.append(list(map(ord,ngram)), str(stress).split(), axis=0)
    record = np.pad(record, pad_width=defaultLength-len(list(map(ord,ngram))),mode='constant')
    total = np.array(record[:defaultLength+1])
    testX = open(dataFile, "a", encoding="utf8")
    testX.write(','.join(total)+"\n")
    testX.close()
    return dataFile
def returnSizedNgram():
    while(True):
        ngram = returnNgrams(data,dictumSize,"random")
        if len(ngram) < defaultLength:
            break
    return ngram
def returnRandomChar():
    letters = string.ascii_lowercase
    length = random.randint(1,8)
    nonWord = ''.join(random.choice(letters) for i in range(length))
    return nonWord
file = returnRandomChar()
while(True):
    resetDataFile(file)
    user = returnRandomChar() + ".dat"
    print()
    print(user)
    recordData(user,0, file)#mode,stress,outputFile
    predict(user,recordData(user,0,file),"stress_model")