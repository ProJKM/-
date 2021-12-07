################ 라이브러리 임포트 ################

import os 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler 
from sklearn.model_selection import train_test_split 
from keras.layers import LSTM 
from keras.layers import Dense 
from keras.layers import Activation
from keras.models import Sequential 
from keras.callbacks import EarlyStopping, ModelCheckpoint 
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

################ 데이터 정규화 ################

# 내림차순으로 데이터 정렬, 기존 행 인덱스 제거후 초기화
df.sort_index(ascending=False).reset_index(drop=True)
# MinMaxScaler 클래스의 인스턴스, 모든 feature가 0과 1사이에 위치하게 만든다.
scaler = MinMaxScaler() 
# 인자 선언
scale_cols = ['시가', '고가', '저가', '종가', '거래량'] 
# df 안의 scale_cols 를 MinMaxScaler
df_scaled = scaler.fit_transform(df [scale_cols]) 
# MinMaxScaler 된 값을 다시 df_scaled 로 선언
df_scaled = pd. DataFrame(df_scaled) 
# df_scaled 의 열을 scale_cols 로 선언
df_scaled.columns = scale_cols 

################ 시계열 데이터셋 분리 ################ 

#### 함수선언

# make_dataset 함수 선언
def make_dataset (data, label, window_size=20):
  # feature_list 선언
  feature_list = [] 
  # label_list 선언
  label_list = [] 
  # data 에서 window_size를 뺀 값 만큼 반복작업
  for i in range(len(data) - window_size):
    # data 에서 i ~ i+window_size 까지 값을 feature_list 에 선언
    feature_list.append(np.array(data.iloc[i:i+window_size]))
    # label 에서 i+window_size 행의 값을 label_list 에 선언
    label_list.append(np.array(label.iloc[i+window_size]))
  # 위에서 나열한 값들을 배열로 변환해, feature_list, label_list 반환
  return np.array(feature_list), np.array(label_list) 

#### 데이터 분리

# TEST_SIZE 선언
TEST_SIZE = 200 
# WINDOW_SIZE 선언
WINDOW_SIZE = 20 
# 최근 200일 제외한 과거데이터로 train 선언
train = df_scaled[:-TEST_SIZE] 
# 최근 200일 데이터로 test 선언
test = df_scaled [-TEST_SIZE:] 
# feature_cols 선언
feature_cols = ['시가', '고가', '저가', '종가', '거래량'] 
# label_cols 선언
label_cols = ['종가'] 

#### 훈련테이터

# train (최근 200일 제외한 데이터) 에서 feature_cols 값 추출해, train_feature 에 선언
train_feature = train[feature_cols]
# train (최근 200일 제외한 데이터) 에서 label_cols 값 추출해, train_feature 에 선언
train_label = train[label_cols] 
# make_dataset 함수를 이용해 반환한 데이터를 train_feature, train_label 에 선언
train_feature, train_label = make_dataset (train_feature, train_label, 20) 
# train_test_split 함수를 이용해 데이터 셔플후 train, test (여기서는 valid) 로 분할
x_train, x_valid, y_train, y_valid = train_test_split(train_feature, train_label, test_size=0.2) 

#### 검증데이터

# test (최근 200일) 에서 feature_cols 값 추출해, test_feature 에 선언
test_feature = test [feature_cols] 
# test (최근 200일) 에서 label_cols 값 추출해, test_label 에 선언
test_label = test [label_cols] 
# make_dataset 함수를 이용해 반환한 데이터를 test_feature, test_label 에 선언
test_feature, test_label = make_dataset (test_feature, test_label, 20) 

################ 모델 학습 ################ 

# 모델 시작
model = Sequential() 
# LSTM, 64유닛, 활성화함수'tanh', 한층더 쌓기위해 return_sequences=True
model.add(LSTM(64,
               input_shape=(train_feature. shape[1], train_feature.shape[2]), 
               activation='tanh',
               return_sequences=True))
# LSTM, 32유닛, return_sequences=False
model.add(LSTM(32,return_sequences=False))
# Dense 레이어
model.add(Dense(1))
# 모델 출력
model.summary()
# 모델 컴파일, 회RNL loss='mse', 최적화 optimizer='Adam'
model.compile(loss='mse', optimizer='Adam')
# early_stop, 3번 연속으로 loss값이 하락하지 않을경우 중단, 과적합 방지
early_stop = EarlyStopping(monitor='loss', patience=3) 
# 모델 학습, epochs = 200회 반복 , batch_size = epoch마다 사용되는 검증데이터 수, validation_data = 검증데이터셋, callbacks = early_stop 적용
model.fit(x_train, y_train,
          epochs=200, 
          batch_size=4, 
          validation_data=(x_valid, y_valid),
          callbacks=[early_stop]) 

# 검증셋으로 예측
pred = model.predict(test_feature) 

################ 시각화 ################ 

# 가로12 세로9의 figure 생성
plt.figure(figsize=(12, 9)) 
# test_label 출력 ,라벨은 'actual'
plt.plot(test_label, label = 'actual') 
# pred 출력 ,라벨은 'prediction'
plt.plot(pred, label = 'prediction') 
# 범례 출력
plt.legend() 

# 학습 평가
score = model. evaluate(x_valid, y_valid, verbose=0)
# loss 값만 출력
print('loss:', score) 
