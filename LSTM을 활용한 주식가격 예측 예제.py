################ 라이브러리 임포트 ################

import numpy as np # linear algebra 
import pandas as pd # data processing, CSV file 1/0 (2.g. pd,read_csv) 
import matplotlib.pyplot as plt 
import seaborn as sns 
from google.colab import drive 

################ 데이터 전처리 ################

# g드라이브 연결
drive.mount('/content/drive') 
# 데이터 로드
stocks = pd.read_csv('/content/drive/MyDrive/01-삼성전자-주가.csv', encoding='utf-8') 
# date컬럼은 자료형이 object이다. 이 문자열 날짜를 datetime 자료형으로 변환
stocks['일자']=pd.to_datetime(stocks['일자'], format='%Y%m%d') 
# 연도 인덱싱
stocks['연도']=stocks['일자'].dt.year 
# 1990년도 이후 자료 인덱싱
df = stocks.loc[stocks['일자']>="1990"] 

################ 시각화 ################

# 가로16 세로9의 figure 생성
plt.figure(figsize=(16, 9)) 
# x축에 일자 y축에 거래량을 넣고 그래프로 출력
sns.lineplot(y=df['거래량'], x=df['일자']) 
# x축 'time' 라벨출력
plt.xlabel('time') 
# y축 'mount' 라벨출력
plt.ylabel('mount') 

################ 데이터 정규화 ################

from sklearn.preprocessing import MinMaxScaler 
# 내림차순으로 데이터 정렬, 기존 행 인덱스 제거후 초기화
df.sort_index(ascending=False).reset_index(drop=True)
# MinMaxScaler 클래스의 인스턴스, 모든 feature가 0과 1사이에 위치하게 만든다.
scaler = MinMaxScaler() 
# 인자 선언
scale_cols = ['시가', '고가', '저가', '종가', '거래량'] 
# df안의 scale_cols를 MinMaxScaler
df_scaled = scaler.fit_transform(df [scale_cols]) 
# MinMaxScaler된 값을 다시 df_scaled로 선언
df_scaled = pd. DataFrame(df_scaled) 
# df_scaled의 열을 scale_cols로 선언
df_scaled.columns = scale_cols 

################ 시계열 데이터셋 분리 ################ 

# TEST_SIZE 선언, 과거 200일 기반으로 학습
TEST_SIZE = 200 
# WINDOW_SIZE 선언, 과거 20일을 기반으로 예측
WINDOW_SIZE = 20 
# 최근 200일 제외한 과거데이터로 훈련셋 선언
train = df_scaled[:-TEST_SIZE] 
# 최근 200일 데이터로 검증셋 선언
test = df_scaled [-TEST_SIZE:] 
# make_dataset 함수 선언
def make_dataset (data, label, window_size=20):
  # feature_list 선언
  feature_list = [] 
  # label_list 선언
  label_list = [] 
  # data에서 window_size를 뺀 값 만큼 반복작업
  for i in range(len(data) - window_size):
    # i ~ i+window_size 까지 값을 feature_list에 선언
    feature_list.append(np.array(data.iloc[i:i+window_size]))
    # i+window_size 행의 값을 label_list에 선언
    label_list.append(np.array(label.iloc[i+window_size]))
  # 위에서 나열한 값들을 배열로 변환해 feature_list, label_list 반환
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

################ 모델 학습 ################ 

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
