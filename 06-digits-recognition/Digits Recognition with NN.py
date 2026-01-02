import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import os
import cv2 as cv

mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = tf.keras.utils.normalize(X_train)
X_test = tf.keras.utils.normalize(X_test)

model = None

if os.path.exists('digits.keras'):
  model = tf.keras.models.load_model('digits.keras')
  print("Model loaded from 'digits.keras'")
  
else:

  model = tf.keras.models.Sequential()
  model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
  model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))
  model.add(tf.keras.layers.Dense(units=128, activation = tf.nn.relu))
  model.add(tf.keras.layers.Dense(units=10, activation='softmax'))

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  model.fit(X_train, Y_train, epochs=3)
  model.save('digits.keras')
  print("Model trained and saved as 'digits.keras'")

loss, accuracy = model.evaluate(X_test, Y_test)

print('Loss: ', loss)
print('Accuracy: ', accuracy)




image = cv.imread('my-image.png')[:,:,0]
image = np.invert(np.array([image]))
image = tf.keras.utils.normalize(image, axis=1)

prediction = model.predict(image)
print ( "Prediction: {}" .format(np.argmax(prediction))) 
plt.imshow(image[0])
plt.show()











