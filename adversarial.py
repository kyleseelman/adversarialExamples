import foolbox
import keras
import numpy as np
from keras import backend
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
from foolbox.attacks import SaliencyMapAttack
from foolbox.criteria import Misclassification
import matplotlib.pyplot as plt

########################################### Loading the model and preprocessing ###############################
backend.set_learning_phase(False)
model = keras.models.load_model('/home/kseelma/VQVAE/WorkingVQVAE/saved_models/keras_mnist_model.h5')
# model = load_model(save_dir + '/keras_mnist_model.h5')
fmodel = foolbox.models.KerasModel(model, bounds=(0,1))
(x_train, y_train),(images, labels) = mnist.load_data()
images = images.reshape(10000,28,28,1)
images= images.astype('float32')
images /= 255

print(images.shape)
print(images[0].shape)
print(labels.shape)
print(labels[0])

print(np.mean(fmodel.forward(images).argmax(axis=-1) == labels))

attackName = "FGSM"

# apply attack on source image
# ::-1 reverses the color channels, because Keras ResNet50 expects BGR instead of RGB
attack = foolbox.attacks.FGSM(fmodel)
adversarials = attack(images[0:650], labels[0:650])
# if the attack fails, adversarial will be None and a warning will be printed

print(np.mean(fmodel.forward(adversarials).argmax(axis=-1) == labels))

print(images[0:32].shape)
print(adversarials.shape)
print(labels.shape)

image = images[0].reshape(28,28)
adversarial = adversarials[0].reshape(28,28)

np.save("imagesfile", images[0:650])
np.save(attackName, adversarials)

plt.figure()

plt.subplot(1, 3, 1)
plt.title('Original')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Adversarial: Uniform Noise')
plt.imshow(adversarial, cmap = 'gray')
plt.axis('off')

#plt.subplot(1, 3, 3)
#plt.title('Difference')
#difference = adversarial - image
#plt.imshow(difference / abs(difference).max() * 0.2 + 0.5, cmap= 'gray')
#plt.axis('off')

plt.show()
