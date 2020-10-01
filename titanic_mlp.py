import numpy as np
import pandas as pd

from scipy.io.arff import loadarff 
from sklearn.model_selection import StratifiedKFold

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.utils import np_utils
from keras.callbacks import CSVLogger
from keras.models import load_model

import matplotlib.pyplot as plt

def training_code():
    #titanic-complete3.arff är som det vi fick men:
    #har attributen normaliserade mellan 0 till 1
    #Filter:ordinal to numeric,
    #cabin och embarked attributet är borttagna

    #Gör om arff filen till numpy arrays.
    raw_data = loadarff('titanic-complete3.arff')
    df_data = pd.DataFrame(raw_data[0])
    arr = df_data.to_numpy()

    #Survival attributet blev en string, fixas här
    numbers = []
    x = 0
    for i in range(arr.size//6):
        for word in arr[x,5].split():
            if word.isdigit():
                numbers.append(int(word))
                x += 1
    y=0
    for i in range(len(numbers)):
        arr[y,5]= numbers[y]
        y+=1

    arr = arr.astype("float32")

    # X_input är de 5 attributen 
    # Y_output är de respektive matchande klasstillhörighetern
    X_input = arr[:,0:5]
    Y_output = arr[:,5]

    num_folds = 10
    acc_per_fold = []
    loss_per_fold = []

    # initialiserar vikterna
    np.random.seed(42)

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    fold_no = 1
    visualo = []
    visuaacc = []
    num_of_epochs = 2000
    #genomför cross validation och tränar modellen
    for train_index, test_index in skf.split(X_input, Y_output):

        # One hot encoding
        X_train, X_test = X_input[train_index], X_input[test_index]
        Y_train, Y_test = np_utils.to_categorical(Y_output[train_index], 2), np_utils.to_categorical(Y_output[test_index], 2)

        #bygger min mlp
        model = Sequential()
        model.add(Dense(4,input_dim = (5), activation="relu"))
        model.add(Dense(2, activation="sigmoid"))

        #Adam är en form av SGD
        model.compile(optimizer="Adam", loss="binary_crossentropy", metrics = ["accuracy"])
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        csv_logger = CSVLogger("training_mlp.log", append=True, separator=";")

        history = model.fit(x=X_train, y=Y_train, epochs=num_of_epochs, callbacks=[csv_logger], verbose=0)

        model.save("Titanic_mlp.h5")
        scores = model.evaluate(X_test, Y_test, verbose=0)
        print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])
        fold_no += 1
        visualo.append(history.history["loss"])
        visuaacc.append(history.history["accuracy"])

    print('------------------------------------------------------------------------')
    print('Score per fold')
    for i in range(0, len(acc_per_fold)):
        print('------------------------------------------------------------------------')
        print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
    print('------------------------------------------------------------------------')
    print('Average scores for all folds:')
    print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
    print(f'> Loss: {np.mean(loss_per_fold)}')
    print('------------------------------------------------------------------------')

    #plottar träningen med snittet för varje epok per k_fold
    herek = np.asarray(visualo, dtype=np.float32)
    loss_visu = np.sum(herek, axis=0)
    plt.plot(loss_visu/num_folds, label='Binary Crossentropy (loss)')
    herek2 = np.asarray(visuaacc, dtype=np.float32)
    acc_visu2 = np.sum(herek2, axis=0)
    plt.plot(acc_visu2/num_folds, label='accuracy/100')
    plt.title('training progress for titanic_mlp')
    plt.ylabel('value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()

def Run():
    nn = load_model("Titanic_mlp.h5")
    pc = int(input("\nförsta, andra eller tredje klass? Svara 1, 2 eller 3: "))
    if pc == 1:
        pc = 0
    if pc == 2:
        pc = 0.5
    if pc == 3:
        pc = 1
    sex = input("\nman eller kvinna?: ")
    if sex == "man":
        sex = 1
    if sex == "kvinna":
        sex = 0
    
    #Varför jag dividerar med så konstiga siffror är för att jag vill likna normalisering från weka som skedde på träningsdatan,
    #där det högsta värdet blev till värdet 1. 
    #en ålder över 80 år skulle alltså ha ett värde mer än ett, men då en av aktiveringsfunktionen är en sigmoid så "löser" det sig hyfsat bra.
    age = int(input("\nvilken ålder? : "))/80
    sibsp = float(input("\nhur många syskon + partner?: "))/8
    fare = int(input("\nBiljettpris i pund? "))/513
    person = np.array([[pc, sex, age, sibsp, fare]])
    person = person.astype("float32")
    prediction = nn.predict(person)
    choice = str(input("se procentuell chans? (svara ja), annars svara nej för död/överlevande output: "))
    if choice == "nej":
        if prediction[0,0] > prediction[0,1]:
            print("död")
        else:
            print("överlevare")

    #i detta sammanhang kan vi tolka klassificerings decimalvärde som procentuell chans
    else:
        if prediction[0,0] > prediction[0,1]:
            print(int(100-prediction[0,0]*100), "procents chans att överleva ")
        else:
            print(int(prediction[0,1]*100), "procents chans att överleva ")