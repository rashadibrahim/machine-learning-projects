from sklearn.svm import SVC
import pickle
import numpy as np
import tensorflow as tf
import os
import cv2 as cv
import numpy as np


mnist = tf.keras.datasets.mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshaping the data to be used in SVC
X_train = X_train.reshape(len(X_train), -1)
X_test = X_test.reshape(len(X_test), -1)

if __name__ == "__main__":
    model = None

    if not os.path.exists('svm_model.pkl'):
        model = SVC(kernel='linear', C=3) 
        model.fit(X_train, Y_train)
        # Save the trained model
        with open('svm_model.pkl', 'wb') as f:
            pickle.dump(model, f)
        print("Model trained and saved as 'svm_model.pkl'")
    else:
        print("Model file already exists. Skipping training....")
        with open('svm_model.pkl', 'rb') as f:
            model = pickle.load(f)
    
    print(f"Model accuracy: {model.score(X_test, Y_test) * 100:.2f}%")




    image = cv.imread('my-image.png')[:,:,0]
    image = np.invert(np.array([image.flatten()]))
    prediction = model.predict(image)
    
    print ( "Prediction without normalization: {}" .format(prediction[0]))
    image = image / 255.0
    prediction = model.predict(image)
    print ( "Prediction with normalization: {}" .format(prediction[0]))









