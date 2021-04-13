import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling1D
from tensorflow.keras.layers import Conv2D, MaxPooling2D,BatchNormalization
import read_datasets

Label = {0:'Fall',1:'Run',2:'Walk',3:'Jog',4:'Jump',5:'up_stair',6:'down_stair',
         7:'stand2sit',8:'sitting',9:'sit2stand'}
CLASS_LIST = [0,2,3,4,5,6,7,9]
CLASS_NUM = len(CLASS_LIST)
print(CLASS_NUM)


x=keras.Input(shape= (20,20,3,),name= 'bitmap_input')
conv1_value= Conv2D(32,(5,5),strides=(1,1),activation='relu',padding="SAME")(x)
mp1= MaxPooling2D(pool_size=(2,2), strides=(1, 1), padding='SAME')(conv1_value)

conv2_value= Conv2D(64,(5,5),strides=(1, 1),activation='relu', padding='SAME')(mp1)
mp2= MaxPooling2D(pool_size=(2,2), strides=(1, 1),padding='SAME')(conv2_value)

flatten= Flatten()(mp2)
fc1= Dense(512 ,activation='relu')(flatten)
dropout=Dropout(rate=0.5)(fc1)
# dropout= BatchNormalization()(fc1)
fc2_output=Dense(8 , activation='relu')(dropout)

model = keras.Model(x,fc2_output,name="FD-CNN")
model.summary()

data = read_datasets.DataSet('..\\data\\dataset',CLASS_LIST)
train_x, train_y = data.get_train_data()
print(train_x)
print(train_y)
train_x= tf.reshape(train_x,[-1,20,20,3])
print(train_x)
train_x= train_x/255.0 * 2 - 1

print(train_x.shape)
model.compile(loss= keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              metrics=["accuracy"])
history = model.fit(train_x , train_y , batch_size= 50 , epochs=1 , validation_split= 0.1)
test_x, test_y = data.get_test_data()
test_x = tf.reshape(test_x , [-1, 20, 20, 3])
test_x = test_x / 255.0 * 2 - 1
print(test_x.shape)
print(test_y.shape)
test_scores= model.evaluate(test_x, test_y, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])

pred= model.predict(test_x)
print(pred.shape)

print(np.argmax(pred,1),np.argmax(test_y,1))


#TFLite Converter
converter= tf.lite.TFLiteConverter.from_keras_model(model)
# Optoimize Optional decrease accuracy
converter.optimizations=[tf.lite.Optimize.DEFAULT]
tflite_model= converter.convert()

#save the TF Lite Model
with tf.io.gfile.GFile('../saved_models/model_v1_0.tflite','wb') as f:
    f.write(tflite_model)