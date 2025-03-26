import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential([
#     tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
#     tf.keras.layers.MaxPooling2D(2, 2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10, activation='softmax')
# ])

# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.keras')

model = tf.keras.models.load_model('/home/user/test/WSI/handwritten.keras')

y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

sensitivity = recall_score(y_test, y_pred_classes, average='macro')
precision = precision_score(y_test, y_pred_classes, average='macro')
accuracy_emnist = accuracy_score(y_test, y_pred_classes)

print(f"Czułość: {sensitivity}")
print(f"Precyzja: {precision}")
print(f"Współczynnik rozpoznawalności: {accuracy_emnist}")

answers = []
for i in range(0, 10):
    for j in range(1, 4):
        filename = f"MyDigits/{i}.{j}.png"
        if os.path.isfile(filename):
            try:
                img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (28, 28))
                img = np.invert(img)
                img = img / 255.0
                img = np.array([img])

                prediction = model.predict(img)
                print(f"Prawdopodobnie jest to cyfra {np.argmax(prediction)}")
                answers.append(np.argmax(prediction))
                plt.imshow(img[0], cmap=plt.cm.binary)
                plt.show()

            except Exception as e:
                print(f"Error przy przetwarzaniu {filename}: {e}")
        else:
            print(f"Brak pliku: {filename}")
correct = 0
count = 0
number = 0
for i in range(len(answers)):
    if answers[i] == number: correct+=1
    count+=1
    if count == 3: 
        number+=1
        count=0
print(f"Współczynnik rozpoznawalności {correct/30}")