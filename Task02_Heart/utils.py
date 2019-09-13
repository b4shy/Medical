import numpy as np
import matplotlib.pyplot as plt


def create_prediction_and_label(prediction):
    values = np.argmax(prediction, axis=4)
    values = np.expand_dims(values, 4)
    return values


def pixel_accuracy(prediction, mask, no_classes=2):
    prediction = np.squeeze(prediction, axis=4)
    sum = 0
    target_sum = np.sum((mask > 0))

    for i in range(1, no_classes):
        target = (mask == i)
        pred = (prediction == i)
        correct_pixels = np.logical_and(target, pred)

        if not np.sum(target) == 0:
            sum += np.sum(correct_pixels) / target_sum

    return sum


def show_prediction_and_label(prediction, label, img):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    fig.show()
    fig.canvas.draw()
    test_prediction = prediction * 10
    prediction2 = np.ma.masked_where(test_prediction == 0, test_prediction)
    orig = np.ma.masked_where(label == 0, label)

    for i in range(80):
        ax.clear()
        ax.imshow(img[0, :, :, i, 0])

        ax.imshow(prediction2[0, :, :, i, 0], alpha=1, cmap="Reds")
        ax.imshow(orig[:, :, i], alpha=0.5, cmap="coolwarm")
        fig.canvas.draw()
    plt.waitforbuttonpress()
