#!/usr/bin/env python
# coding: utf-8

# In[24]:


# get_ipython().system('pip uninstall tensorflow numpy -y')


# In[25]:


# get_ipython().system('pip install numpy==1.24')


# In[26]:


# get_ipython().system('pip install tensorflow==2.10.0')


# In[ ]:





# In[27]:


import numpy as np
import tensorflow as tf
from tensorflow import keras


# In[56]:


model = tf.keras.models.load_model('xception_v2_13_0.865.keras')
model


# In[57]:


get_ipython().system('wget "https://storage.googleapis.com/kagglesdsdata/datasets/1608934/2645886/Training/glioma/Tr-glTr_0005.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20250115%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250115T112321Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=8f96ece58744acf1e3dadab607f81c41f7724cff7ba628cfff3a5ed7b3c0ca782dc87695b5d99c57e290a28ae4abd6bcc521473ab19819de35a9726e002198b12fb508a3582f58c55623607b4a8aa3dac032eae11447d932f4a4595c7e0c8556b06b4ba784c9de6b329115718051e0d2fb8fceb017b0bef782af1b23bc17493a79a24ad9bb0ad97f130a3009b99a4d6c6faa077bd06a1735088f0a337bf862ba657d30119bff0bb78cba87bfc2f670dd51de1aba76b0f0c30498612ecefb5cc9fd1640bb66dfc6c2d34c9b5abbce1a8066aeeecd9b13020162e446ff8a7812e2aef9d430ffd6d204861c4a9972dda6c11a2be7336e012944a589f20bf70bbb6a" -O glioma_image.jpg')


# In[58]:


from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.xception import preprocess_input


# In[59]:


image=load_img('glioma_image.jpg', target_size=(299,299))
image


# In[60]:


x=np.array(image)
X=np.array([x])

X=preprocess_input(X)
X


# In[61]:


X.shape


# In[62]:


import numpy as np

dummy_input = np.random.rand(1, *model.input_shape[1:])  # Create dummy data matching input shape
model.predict(dummy_input)


# In[63]:




preds=model.predict(X)
preds


# In[66]:


classes=['glioma','meningioma', 'notumor', 'pituitary']


# In[67]:


dict(zip(classes, preds[0]))


# # Converting Tensorflow model to Tensorflow Lite

# In[68]:


converter=tf.lite.TFLiteConverter.from_keras_model(model)


# In[72]:


tflite_model=converter.convert()


# In[73]:


with open('brain-tumour-model.tflite', 'wb') as f_out:
    f_out.write(tflite_model)


# In[75]:


get_ipython().system('ls')


# In[76]:


import tensorflow.lite as tflite


# In[77]:


interpreter=tflite.Interpreter(model_path='brain-tumour-model.tflite')
interpreter.allocate_tensors()


# In[92]:


input_index=interpreter.get_input_details()[0]['index']
output_index=interpreter.get_output_details()[0]['index']


# In[93]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds=interpreter.get_tensor(output_index)


# # Removing tensorflow dependacy

# In[ ]:


from PIL import Image


# In[94]:


with Image.open ('glioma_image.jpg') as img:
    img=img.resize((299,299), Image.NEAREST)


# In[95]:


img


# In[99]:


def preprcess_input(X):
    x/=127.5
    x-=1.
    return x


# In[100]:



x=np.array(img, dtype='float32')
X=np.array([x])

X=preprocess_input(X)
X


# In[101]:


interpreter.set_tensor(input_index, X)
interpreter.invoke()
preds=interpreter.get_tensor(output_index)


# In[102]:


classes=['glioma','meningioma', 'notumor', 'pituitary']
dict(zip(classes, preds[0]))
















