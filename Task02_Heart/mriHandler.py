import os
import numpy as np
from nilearn import image
from imgaug import augmenters as iaa
import imgaug as ia


class MRIHandler():

    INPUT_SHAPE = (320, 320, 80)
    LAST_CHANNEL_NO = INPUT_SHAPE[2]
    ROTATION_PARAMS = np.array([0, 45, 90, 135, 180, 225, 270])

    IMG_PATH = "imagesTr/"
    LABEL_PATH = "labelsTr/"

    def __init__(self):

        self.paths_to_all_imgs = self._create_path_list(self.IMG_PATH)
        self.paths_to_all_labels = self._create_path_list(self.LABEL_PATH)
        self.batch_size = 2
        self.classes = 2

    def next_batch(self, batch_size=2):
        self.batch_size = batch_size
        ind = np.random.choice(self.paths_to_all_imgs, batch_size)
        imgs, labels = self._load_batch(ind)

        aug_img, aug_labels = self.augment(imgs, labels)

        label_mask = self._create_one_hot_label(aug_labels)

        return aug_img, label_mask, aug_labels

    def augment(self, imgs, labels):
        index = np.random.randint(0, 4, size=self.batch_size)
        actual_rotation = self.ROTATION_PARAMS[index]

        imgs = np.squeeze(imgs, -1)
        aug_imgs = []
        aug_labels = []

        for i in range(len(actual_rotation)):
            rotate = iaa.Affine(rotate=(actual_rotation[i]))

            aug_imgs.append(rotate.augment_image(imgs[i]))
            aug_labels.append(rotate.augment_image(labels[i]))

        return np.expand_dims(aug_imgs, -1), np.array(aug_labels)

    def _load_batch(self, ind):
        imgs = []
        labels = []
        for i in ind:
            s = i.split("/")
            img = image.smooth_img(i, None).get_data()[:, :, :self.LAST_CHANNEL_NO]
            imgs.append(img)

            label_path_name = f'labelsTr/{s[1]}'
            label = image.smooth_img(label_path_name, None).get_data()[:, :, :self.LAST_CHANNEL_NO]
            labels.append(label)
        return np.expand_dims(np.array(imgs), -1), np.array(labels)

    def _create_one_hot_label(self, label_mask):
        mask = np.zeros((self.batch_size, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], self.INPUT_SHAPE[2],
                         self.classes))

        for c in range(self.batch_size):
            actual_label_mask = label_mask[c]
            for i in range(self.classes):
                equal = np.equal(actual_label_mask, i)
                mask[c, :, :, :, i] = equal

        return mask

    def _create_path_list(self, path):
        return_list = []
        for i in os.listdir(path):
            temp_path = os.path.join(path, i)
            return_list.append(temp_path)
        return return_list

    def load_val_data(self, val_path):
        imgs = []
        labels = []
        val_path_imgs = os.path.join(val_path, "imgs")
        val_path_labels = os.path.join(val_path, "labels")

        for i in os.listdir(val_path_imgs):
            img_path = os.path.join(val_path_imgs, i)
            img = image.smooth_img(img_path, None).get_data()[:, :, :self.LAST_CHANNEL_NO]
            imgs.append(img)

            label_path = os.path.join(val_path_labels, i)
            label = image.smooth_img(label_path, None).get_data()[:, :, :self.LAST_CHANNEL_NO]
            labels.append(label)
        return np.expand_dims(np.array(imgs), -1), np.array(labels)


