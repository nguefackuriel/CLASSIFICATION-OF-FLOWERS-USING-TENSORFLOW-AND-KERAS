#!/usr/bin/env python
# coding: utf-8

# In[32]:


from google.colab import drive
drive.mount('/content/drive')


# In[33]:


get_ipython().run_line_magic('matplotlib', 'inline')

import time
import PIL
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
tfds.disable_progress_bar()


# In[34]:


# Load model
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224,3))


# In[35]:


feature_extractor.trainable = False


# In[36]:


model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Dense(102, activation="softmax")
])
model.summary()


# In[37]:


model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"] )


# In[38]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale=1/255)
valid_datagen = ImageDataGenerator(rescale=1/255)



train_generator = train_datagen.flow_from_directory(
    os.path.join('/content/drive/MyDrive/flower/train'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(
    os.path.join('/content/drive/MyDrive/flower/valid'),
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary')


# In[39]:


class_names = train_ds.class_names
class_names1 = valid_ds.class_names
print(class_names)
print(class_names1)


# In[40]:


data_dir = os.path.join('/content/drive/MyDrive/flower/train')
valid_dir = os.path.join('/content/drive/MyDrive/flower/valid')

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir)
valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
  valid_dir)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")


# In[61]:


EPOCHS = 8

history = model.fit(train_generator, epochs=EPOCHS, validation_data=validation_generator)


# In[15]:


image_size = 224

def process_image(image):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return image


# In[16]:


from PIL import Image
# TODO: Create the predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    
    processed_image = process_image(image)
    processed_image = np.expand_dims(processed_image,axis=0)
    ps = model.predict(processed_image)
    probs = ps[0]
    
    # Get k indexes of sorted probs
    top_indexes = np.argsort(probs)[-top_k:]
    
    # Reverse array
    top_indexes = top_indexes[::-1]
    
    #top_probs = probs.sort()[-top_k:]
    probs.sort()
    probs = probs[-top_k:][::-1]
    return (probs,top_indexes)


# In[30]:


from PIL import Image
def show_image(image_path, label, prob=""):
    image = PIL.Image.open(image_path)
    image = np.asarray(image)
    image = process_image(image)

    plt.imshow(image)
    plt.title(str(label) +"  " + str(prob))
    plt.show()
    return


# In[67]:


from PIL import Image
image_path = '/content/drive/MyDrive/flower/marguerite-729510_1920.jpg'
(probs, classes) = predict(image_path, model, 5)

show_image(image_path, class_names[int(classes[0])], probs[0])


# In[68]:


tf.saved_model.save(model,'/content/drive/MyDrive/flower')


# In[66]:


# TODO: Plot the loss and accuracy values achieved during training for the training and validation set.
training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

