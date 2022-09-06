import  numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# store image in the same folder as this file
# and just run
#and input image name with file format in terminal
#png may not work
# don't do anything else


def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s


def predict(w, b, X):
    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)
    
    for i in range(A.shape[1]):
        if A[0, i] > 0.7:
            Y_prediction[0, i] = A[0, i]
        else:
            Y_prediction[0, i] = A[0, i]
    return Y_prediction

my_image = input('Input the name of image to test \n')



im = Image.open(my_image)
#im.show()

image = np.array(Image.open(my_image).resize((64, 64)))
image = image / 255
image = image.reshape((1, 64 * 64 * 3)).T

w = np.loadtxt('W.csv', delimiter=',')
b = -0.003290923362980046

prediction = predict(w, b, image)



print(f'resemblance: {prediction * 100} % (percent)')
if prediction > 0.6:
    print('It is most likely a dog')
else:
    print('I doubt it\'s a dog')