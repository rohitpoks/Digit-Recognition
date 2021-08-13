import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pygame as pg
import tensorflow as tf


def init():
    global screen

    pg.init()
    screen = pg.display.set_mode((140, 140))
    mainloop()


drawing = False
last_pos = None
w = 5
color = (255, 255, 255)


def draw(event):
    global drawing, last_pos, w

    if event.type == pg.MOUSEMOTION:
        if drawing:
            mouse_position = pg.mouse.get_pos()
            if last_pos is not None:
                pg.draw.line(screen, color, last_pos, mouse_position, w)
            last_pos = mouse_position
    elif event.type == pg.MOUSEBUTTONUP:
        mouse_position = (0, 0)
        drawing = False
        last_pos = None
    elif event.type == pg.MOUSEBUTTONDOWN:
        drawing = True


def mainloop():
    global screen

    loop = 1
    while loop:

        for event in pg.event.get():
            if event.type == pg.QUIT:
                loop = 0
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_s:
                    pg.image.save(screen, "image.png")
            draw(event)
        pg.display.flip()
    pg.quit()


def convert():
    img = Image.open('image.png')
    array = np.array(img)
    print(array)
    print(array.shape)
    temp_list = []
    array_temp1 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                   [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    for i in range(140):
        for j in range(140):
            if array[i, j, 0] == 255:

                array_temp1[i].append(1)

            else:
                array_temp1[i].append(0)
    #  converting to 28x28
    te_pixels = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [],
                 [], []]
    for i in range(28):
        for j in range(28):
            te_pixels[i].append((array_temp1[i * 5][j * 5] + array_temp1[i * 5][j * 5 + 1] + array_temp1[i * 5][
                j * 5 + 2] + array_temp1[i * 5][j * 5 + 3] + array_temp1[i * 5][j * 5 + 4] + array_temp1[i * 5 + 1][
                                     j * 5] + array_temp1[i * 5 + 1][j * 5 + 1] + array_temp1[i * 5 + 1][j * 5 + 2] +
                                 array_temp1[i * 5 + 1][j * 5 + 3] + array_temp1[i * 5 + 1][j * 5 + 4] +
                                 array_temp1[i * 5 + 2][j * 5] + array_temp1[i * 5 + 2][j * 5 + 1] +
                                 array_temp1[i * 5 + 2][j * 5 + 2] + array_temp1[i * 5 + 2][j * 5 + 3] +
                                 array_temp1[i * 5 + 2][j * 5 + 4] + array_temp1[i * 5 + 3][j * 5] +
                                 array_temp1[i * 5 + 3][j * 5 + 1] + array_temp1[i * 5 + 3][j * 5 + 2] +
                                 array_temp1[i * 5 + 3][j * 5 + 3] + array_temp1[i * 5 + 3][j * 5 + 4] +
                                 array_temp1[i * 5 + 4][j * 5] + array_temp1[i * 5 + 4][j * 5 + 1] +
                                 array_temp1[i * 5 + 4][j * 5 + 2] + array_temp1[i * 5 + 4][j * 5 + 3] +
                                 array_temp1[i * 5 + 4][j * 5 + 4]) / 25)
    print(te_pixels)
    return te_pixels


def nn():
    import tensorflow as tf
    import numpy as np
    mnist = tf.keras.datasets.mnist  # 28x28 images of digits 0-9
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=3)
    val_los, val_acc = model.evaluate(x_test, y_test)
    print(val_los, val_acc)
    model.save('num_digit_reader_model')
    new_model = tf.keras.models.load_model('num_digit_reader_model')


nn()

init()
new_model = tf.keras.models.load_model('num_digit_reader_model')
predictions = new_model.predict([convert()])
imgplot = plt.imshow(convert())
print('-------------------------------------------')
print('i think the number is:')
print(np.argmax(predictions))
