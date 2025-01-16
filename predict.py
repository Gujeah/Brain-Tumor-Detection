#!/usr/bin/env python
# coding: utf-8




import wget
import tensorflow as tf
from tensorflow import keras
from tensorflow import keras
import tensorflow.lite as tflite
from PIL import Image
import numpy as np
import tensorflow as tf
load_img = keras.utils.load_img
img_to_array = keras.utils.img_to_array
Xception = keras.applications.Xception
preprocess_input = keras.applications.xception.preprocess_input
# from tensorflow.keras.preprocessing.image import load_img
# from tensorflow.keras.applications.xception import preprocess_input



model = tf.keras.models.load_model('xception_v2_13_0.865.keras')
model


# url = "https://storage.googleapis.com/kagglesdsdata/datasets/1608934/2645886/Training/glioma/Tr-glTr_0005.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250115%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250115T112321Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=8f96ece58744acf1e3dadab607f81c41f7724cff7ba628cfff3a5ed7b3c0ca782dc87695b5d99c57e290a28ae4abd6bcc521473ab19819de35a9726e002198b12fb508a3582f58c55623607b4a8aa3dac032eae11447d932f4a4595c7e0c8556b06b4ba784c9de6b329115718051e0d2fb8fceb017b0bef782af1b23bc17493a79a24ad9bb0ad97f130a3009b99a4d6c6faa077bd06a1735088f0a337bf862ba657d30119bff0bb78cba87bfc2f670dd51de1aba76b0f0c30498612ecefb5cc9fd1640bb66dfc6c2d34c9b5abbce1a8066aeeecd9b13020162e446ff8a7812e2aef9d430ffd6d204861c4a9972dda6c11a2be7336e012944a589f20bf70bbb6a"
# output = "glioma_image.jpg"

image = load_img("glioma.jpg", target_size=(299, 299))






x=np.array(image)
X=np.array([x])

X=preprocess_input(X)
X



X.shape




dummy_input = np.random.rand(1, *model.input_shape[1:]) 
model.predict(dummy_input)


preds=model.predict(X)
preds



classes=['glioma','meningioma', 'notumor', 'pituitary']




dict(zip(classes, preds[0]))


# # Converting Tensorflow model to Tensorflow Lite



converter=tf.lite.TFLiteConverter.from_keras_model(model)



tflite_model=converter.convert()



with open('brain-tumour-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)



interpreter=tflite.Interpreter(model_path='brain-tumour-model.tflite')
interpreter.allocate_tensors()

input_index=interpreter.get_input_details()[0]['index']
output_index=interpreter.get_output_details()[0]['index']

interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds=interpreter.get_tensor(output_index)


# # Removing tensorflow dependacy

with Image.open ('glioma_image.jpg') as img:
    img=img.resize((299,299), Image.NEAREST)
img


def preprcess_input(X):
    x/=127.5
    x-=1.
    return x


x=np.array(img, dtype='float32')
X=np.array([x])

X=preprocess_input(X)
X


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds=interpreter.get_tensor(output_index)


classes=['glioma','meningioma', 'notumor', 'pituitary']
dict(zip(classes, preds[0]))
















