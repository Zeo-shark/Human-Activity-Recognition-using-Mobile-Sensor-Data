import numpy as np
import tensorflow as tf
import pandas as pd
import time

start= time.time()

#Load the TFLite and allocate the input tensors
interpreter= tf.lite.Interpreter(model_path='model_stride1.tflite')
interpreter.allocate_tensors()

# Get the Input and Output Tensor
input_details=interpreter.get_input_details()
output_details=interpreter.get_output_details()

#Test the model on Random Input Data
input_shape=input_details[0]['shape']
print(input_details[0])
input_data= np.array(np.random.random_sample(input_shape), dtype=np.float32)

interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

output_data= interpreter.get_tensor( output_details[0]['index'])
print(output_details[0])
out=np.argmax(output_data,1)
print(input_data)
print(output_data)

print(out)

data=pd.read_csv('../data/test.csv')
test_x=[]
test_x.append(data.iloc[1,1:1201])

test_x= tf.reshape(test_x,input_shape)
test_x= tf.cast(test_x, dtype=tf.float32)
test_x= test_x/255.0 * 2 - 1
print(test_x)

interpreter.set_tensor(input_details[0]['index'], test_x)
interpreter.invoke()
output=interpreter.get_tensor(output_details[0]['index'])
print(output)

print('elapsed',(time.time()- start))

