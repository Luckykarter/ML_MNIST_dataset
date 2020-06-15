import tensorflow as tf
import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time

# if we want to stop training at some point - there is callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        acc = 0.9
        if(logs.get('accuracy') >= acc):
            print('\nReached ' + str((1-acc) * 100) +'% accuracy so cancelling training')
            self.model.stop_training = True

callbacks = myCallback()

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()

# print(training_labels[0])
# print(training_images[0])
#
# plt.imshow(training_images[0])
# plt.show()

# if we training a neural network - it is easier
# with values from 0 to 1
# this process called normalizing

# divide entire arrays
training_images= training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images = training_images / 255.0
test_images = test_images / 255

model = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
                                    tf.keras.layers.MaxPooling2D(2, 2),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
# the number of neurons in the last layer should match the
# number of classes you are classifying for

# Sequential - defines a SEQUENCE of layers in the neural network
# Flatten - flattens the square (make 1-dimensional set)
# Dense - adds a layer of neurons
#          activation_function tells them what to do
# Relu - "if X>0 return X else return 0" passes 0 or greater to the next layer
# Softmax - takes a set and picks the biggest one.

model.compile(optimizer=tf.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])
# train network
tic = time.perf_counter()
model.fit(training_images, training_labels, epochs=1000,
          callbacks=[callbacks])
toc = time.perf_counter()
print("Time spent: " + str(toc - tic))

print("evaluate on unseen data:")
model.evaluate(test_images, test_labels)

print("exercise 1")
classifications = model.predict(test_images)
# the value is the probability that the item is each of the 10 classes
print([str(round(x*100, 3)) + "%" for x in classifications[0]])