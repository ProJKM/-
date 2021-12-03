import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file 1/0 (2.g. pd,read_csv) 
import matplotlib.pyplot as plt 
import seaborn as sns 
from google.colab import drive 
# g드라이브 연결
drive.mount('/content/drive') 
# 데이터 로드
stocks = pd.read_csv('/content/drive/MyDrive/01-삼성전자-주가.csv', encoding='utf-8') 
# 
stocks['일자']=pd.to_datetime(stocks['일자'], format='%Y%m%d') 
stocks['연도']=stocks['일자'].dt.year 
df = stocks.loc[stocks['일자']>="1990"] 
plt.figure(figsize=(16, 9)) 
sns.lineplot(y=df['거래량'], x=df['일자']) 
plt.xlabel('time') 
plt.ylabel('mount') 

from sklearn.preprocessing import MinMaxScaler 
df.sort_index(ascending=False).reset_index(drop=True) 
scaler = MinMaxScaler() 
scale_cols = ['시가', '고가', '저가', '종가', '거래량'] 
df_scaled = scaler.fit_transform(df [scale_cols]) 
df_scaled = pd. DataFrame(df_scaled) 
df_scaled.columns = scale_cols 
df_scaled 
TEST_SIZE = 200 
WINDOW_SIZE = 20 
train = df_scaled[:-TEST_SIZE] 
test = df_scaled [-TEST_SIZE:] 
def make_dataset (data, label, window_size=20):
  feature_list = [] 
  label_list = [] 
  for i in range(len(data) - window_size):
    feature_list.append(np.array(data.iloc[i:i+window_size]))
    label_list.append(np.array(label.iloc[i+window_size]))
  return np.array(feature_list), np.array(label_list) 
from sklearn.model_selection import train_test_split 
feature_cols = ['시가', '고가', '저가', '종가', '거래량'] 
label_cols = ['종가'] 
train_feature = train[feature_cols]
train_label = train[label_cols] 
train_feature, train_label = make_dataset (train_feature, train_label, 20) 
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2) 
x_train.shape, x_valid.shape
test_feature = test [feature_cols] 
test_label = test [label_cols] 
test_feature.shape, test_label.shape 
test_feature, test_label = make_dataset (test_feature, test_label, 20) 
test_feature. shape, test_label.shape

from keras.models import Sequential 
from keras.layers import Dense 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.layers import LSTM 
model = Sequential() 
model.add(LSTM(32,
               input_shape=(train_feature. shape[1], train_feature.shape[2]), 
               activation='softmax', 
               return_sequences=False)
)
model.add(Dense(1))
import os 
model.compile(loss='mean_squared_error', optimizer='RMSprop')
early_stop = EarlyStopping(monitor='loss', patience=1) 
history = model.fit(x_train, y_train,
                    epochs=30, 
                    batch_size=16, 
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stop]) 
pred = model.predict(test_feature) 
pred.shape 
plt.figure(figsize=(12, 9)) 
plt.plot(test_label, label = 'actual') 
plt.plot(pred, label = 'prediction') 
plt.legend() 
plt.show() 
score = model. evaluate(x_valid, y_valid, verbose=1)
