############ 변경점 ############

# ImageDataGenerator를 이용하여 밝기조절 이미지 추가 생성 >> 훈련표본 증가
# 기본 모델에서 VGG-16 모델로 변경 >> 레이어 증가
# 활성함수(Activation Function) relu >> swish
# 배치 정규화(Batch Normalization) 추가, 드랍아웃(Dropout)은 사용안함
# 최적화(Optimization) adadelta >> Adam
# ILSVRC 기준 15만장, 256 batch_size, 훈련량 585 >> 5216장 기준 동일 훈련량일시 batch_size = 9 로 설정
# epochs 10 >> 30 

############ 라이브러리 로드 ############

from google.colab import auth 
auth.authenticate_user()
from google.colab import drive 
drive.mount('/content/gdrive') 
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 
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
from keras.layers import BatchNormalization, Dropout

################ 데이터 셋 전처리 ################

# 시드 고정
np.random.seed(3)

# ImageDataGenerator 클래스 함수화
# 훈련셋 밝기조절 이미지 추가
train_datagen = ImageDataGenerator(brightness_range=[0.2,1.0])
test_datagen = ImageDataGenerator()

# 데이터셋 임포트
# ImageDataGenerator 클래스에서 전처리가 끝남
# 사이즈 고정, 가용램에 따라 batch_size 조절
train_generator = train_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/chest_xray/train', 
    target_size=(224, 224), 
    batch_size=5211, 
    class_mode='categorical')
test_generator = test_datagen.flow_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/chest_xray/test', 
    target_size=(224, 224), 
    batch_size=621, 
    class_mode='categorical')

# 데이터를 함수에 정의
x_train, y_train=train_generator.next() 
x_test, y_test=train_generator.next()

################ 모델 구축 및 학습 ################

# 학습 모델, 기본뼈대 VGG-16
# 활성함수(Activation Function) relu >> swish
# 최적화(Optimization) adadelta >> Adam
# 배치 정규화(Batch Normalization) 추가
model = Sequential()
# 컨볼루션 레이어1, 컨볼루션층, 64개 필터, 커널 사이즈는 (3,3)고정, 활성화 함수는 'swish'
model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="swish"))
# 컨볼루션층
model.add(Conv2D(filters=64, kernel_size=(3,3),padding="same"))
# 배치정규화
model.add(BatchNormalization()) 
# 활성화 함수 swish
model.add(Activation('swish'))
# 풀링층, 맥스풀링
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# 컨볼루션 레이어2, 128개 필터
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=128, kernel_size=(3,3),padding="same"))
model.add(BatchNormalization()) 
model.add(Activation('swish'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# 컨볼루션 레이어3, 256개 필터
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=256, kernel_size=(3,3),padding="same"))
model.add(BatchNormalization()) 
model.add(Activation('swish'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# 컨볼루션 레이어4, 512개 필터
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=512, kernel_size=(3,3),padding="same"))
model.add(BatchNormalization()) 
model.add(Activation('swish'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# 컨볼루션 레이어5, 512개 필터
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="swish"))
model.add(Conv2D(filters=512, kernel_size=(3,3),padding="same"))
model.add(BatchNormalization()) 
model.add(Activation('swish'))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
# FC
model.add(Flatten())
model.add(Dense(units=4096,activation="swish"))
model.add(Dense(units=4096,activation="swish"))
# softmax로 결론도출
model.add(Dense(units=2, activation="softmax"))
# 모델 가시화
model.summary()

# 모델 컴파일, loss, optimizer, metrics를 설정 
# optimizer는 역전파 진행시 최적화 방식, 여기서는 Adam 사용
# loss는 손실함수, 데이터를 검증하는 방식
# metrics의 경우 분류문제이기 때문에 accuracy
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

# epochs = 반복 횟수
epochs = 30
# batch_size = 가중치를 갱신할 데이터의 양
batch_size = 9

# 3번 연속으로 loss값이 하락하지 않을경우 중단, 과적합 방지
early_stopping = EarlyStopping(monitor='loss', patience=3) 

# 모델 학습
# verbose = 학습 진행도 노출설정, validation_data = 검증데이터셋, callbacks = epoch마다 사용되는 검증데이터 수
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test), callbacks=[early_stopping])

################ 결과 ################

# 학습 검증, 평가
score = model.evaluate(x_test, y_test, verbose=0) 

# 출력
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

# 결과 시각화를 위한 함수화
op=[]
# 나열
op.append(history)

# 시각화
# loos를 가져옴, 훈련셋
plt.plot(op[0].history['loss'])
# val_loos를 가져옴, 검증셋
plt.plot(op[0].history['val_loss'])
# y축
plt.ylabel('loss')
# x축
plt.xlabel('epoch')
# 범례추가, loc = location 범례 위치
plt.legend(['train', 'val'], loc='upper left')
