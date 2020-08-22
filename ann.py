from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_dataset(title):
    return pd.read_csv('./'+title+'.csv')

# Load data ignoring Rownumber, Customer ID and Surname
dataframe = load_dataset('Churn_Modelling')

def set_inputs(range_x_1, range_x_2):
    return dataset[:,range_x_1:range_x_2]

def set_output(range_y):
    return dataset[:,range_y]

# Transform Geography and Gender to numerical values
le = preprocessing.LabelEncoder()
encoded = dataframe.apply(le.fit_transform)
dataset = encoded.values

# set inputs - output
X = set_inputs(range_x_1 = 3, range_x_2 = 13)
y = set_output(range_y = 13)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Rescale min and max for X
scaler = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler.fit_transform(X)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Kerasa sıralı bir modelimizin olduğunu belirtiyoruz.
# Eklenen her katmanın çıktısı kendinden bir sonraki katmana girdi olacağını belirtiyoruz.
model = Sequential()

# sinir ağımıza katman eklemek için .add kullanıyoruz.
# Dense'i katmanların bağlı olduğunu belirtmek için kullanıyoruz.
# Dense: katman boyutu, input_dim: girdi boyutu, activation: kullandığımız aktivasyon fonksiyonu.
model.add(Dense(16, input_dim=10, activation='relu'))
# Sıralı olduğunu belirttiğimiz için tekrar girdi boyudu vermemiz gerekmez.
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train , validation_data = (X_test,y_test) , epochs=100, batch_size=50)

# Print Accuracy
scores = model.evaluate(rescaledX, y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

#plot
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
