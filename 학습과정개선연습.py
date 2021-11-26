# -*- coding: utf-8 -*-
"""2번코드.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1IdqwLCIR1Ow9f8MK1TiM-IzWVQasXA0i
"""

############ 라이브러리 로드 ############

from google.colab import auth 
auth.authenticate_user()
from google.colab import drive 
drive.mount('/content/gdrive') 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as pit 
from tensorflow.keras.preprocessing import sequence 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, Activation 
from tensorflow.keras.layers import LSTM 
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, GlobalMaxPooling2D, Bidirectional, Reshape
from tensorflow.keras.layers import Flatten 
from tensorflow.keras.layers import TimeDistributed 
from tensorflow.keras.layers import RepeatVector 
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
np.random.seed(3)

################ 데이터 셋 전처리 ################

# ImageDataGenerator 클래스 함수화
train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

# 데이터셋 임포트
# ImageDataGenerator 클래스에서 전처리가 끝남
train_generator = train_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/chest_xray/train', 
    target_size=(224, 224), 
    batch_size=5216, 
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/chest_xray/test', 
    target_size=(224, 224), 
    batch_size=624, 
    class_mode='categorical')

# 전처리가 끝난 데이터를 함수에 정의
x_train, y_train=train_generator.next() 
x_test, y_test=train_generator.next()

################ 모델 구축 및 학습 ################

# 학습 모델
model = Sequential()
# 컨볼루션 레이어, 컨볼루션층, 32개의 필터, 커널사이즈는 (3, 3), 픽셀은 그대로, 활성화 함수는 relu
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224,3))) 
# 풀링층, 방식은 맥스풀링
model.add(MaxPooling2D(pool_size=(2, 2))) 
# 컨볼루션 레이어 한번더, 필터는 64개
model.add(Conv2D(64, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
# Flatten, 모델 안에서의 reshape, 한줄로 변형
model.add(Flatten()) 
# 결과, 활성화 함수는 softtmax
model.add(Dense(2, activation='softmax'))
# 모델구조 확인
model.summary()

# 모델 컴파일, loss, optimizer, metrics를 설정 
# optimizer는 역전파 진행시 최적화 방식
# loss는 손실함수, 데이터를 검증하는 방식
# metrics의 경우 분류문제이기 때문에 accuracy
model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy']) 
# epochs = 반복 횟수
epochs = 10 
# batch_size = 가중치를 갱신할 데이터의 양
batch_size=100

# 3번 연속으로 loss값이 하락하지 않을경우 중단, 과적합 방지
early_stopping = EarlyStopping(monitor='loss', patience=3) 
# 모델 학습
# verbose = 학습 진행도 노출설정, validation_data = 검증데이터셋, callbacks = epoch마다 사용되는 검증데이터 수
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])
# 결과 시각화를 위한 함수화
op=[]
# 나열
op.append(history)

################ 결과 ################

# 학습 검증, 평가
score = model.evaluate(x_test, y_test, verbose=0) 

# 출력
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# 시각화
# loos를 가져옴, 훈련셋
plt.plot(op[0].history['loss'])
# val_loos를 가져옴, 검증셋
plt.plot(op[0].history['val_loss'])
# 제목
plt.title('{}'.format(optimizers[0]))
# y축
plt.ylabel('loss')
# x축
plt.xlabel('epoch')
# 범례추가, loc = location 범례 위치
plt.legend(['train', 'val'], loc='upper left')
