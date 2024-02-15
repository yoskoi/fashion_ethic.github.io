#Les package
import tensorflow as  tf # pip intall tensorflow
from tensorflow.keras import datasets, layers, models # pip install keras
import matplotlib.pyplot as plt # pip install matplotlib
from PIL import Image # package integre dans python qui nous convetis les images en binaire
import numpy as np # cet package nous permet de transformer nos image en binaire en un matrics

# Chargement les données de Fashion MNIST dataset integre dans Keras
(train_images, train_labels), (test_images, test_labels) = datasets.fashion_mnist.load_data()

# Normaliseons les images
train_images, test_images = train_images / 255.0, test_images / 255.0

# Créeons une modèle sequentiel simple avec 10 neuron 
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(32, activation='relu'),
    layers.Dense(10)
])

# nous  Compileons le modèle avec optizer adam et afficher les matics avec accuracy
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# on entraine notre model avec model fit Entraîner le modèle
model.fit(train_images, train_labels, epochs=500)

# Faire une prédiction
predictions = model.predict(test_images)
print(predictions[50])


plt.imshow(train_images[0])
print(train_labels[0])
print(train_images[0])
