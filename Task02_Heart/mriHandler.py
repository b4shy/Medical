import os
import numpy as np
from nilearn import image
from imgaug import augmenters as iaa


class MRIHandler():
    INPUT_SHAPE = (320, 320, 128)
    LAST_CHANNEL_NO = INPUT_SHAPE[2]

    IMG_PATH = "imagesTr/"
    LABEL_PATH = "labelsTr/"

    def __init__(self):

        self.epoch_counter = 0
        self.paths_to_all_imgs = self._create_path_list(self.IMG_PATH)
        self.paths_to_all_labels = self._create_path_list(self.LABEL_PATH)
        self.epoch_path_list = []
        self.batch_size = 2
        self.classes = 2

    def next_batch(self, batch_size=2):
        self.batch_size = batch_size

        self.epoch_handler()

        ind = self._pop_next_batch()

        imgs, labels = self._load_batch(ind)

        aug_img, aug_labels = self.augment(imgs, labels)

        label_mask = self._create_one_hot_label(aug_labels)

        return np.expand_dims(aug_img, -1), label_mask, aug_labels

    def epoch_handler(self):
        if len(self.epoch_path_list) < self.batch_size:
            self._prepare_next_epoch()
            self.epoch_counter += 1
            print(f'##########Epoch: {self.epoch_counter}##########')

    def _pop_next_batch(self):
        ind = []

        for i in range(self.batch_size):
            if len(self.epoch_path_list) > 0:
                ind.append(self.epoch_path_list.pop(0))
        return ind

    def _prepare_next_epoch(self):
        np.random.shuffle(self.paths_to_all_imgs)
        self.epoch_path_list = self.paths_to_all_imgs.copy()

    def _load_batch(self, ind):
        imgs = []
        labels = []
        for i in ind:
            s = i.split("/")
            img = image.smooth_img(i, None).get_data()
            img = self.maybe_pad(img)
            img = img[:, :, :self.LAST_CHANNEL_NO]
            imgs.append(img)

            label_path_name = f'labelsTr/{s[1]}'
            label = image.smooth_img(label_path_name, None).get_data()
            label = self.maybe_pad(label)
            label = label[:, :, :self.LAST_CHANNEL_NO]
            labels.append(label)
        return np.array(imgs), np.array(labels)

    def maybe_pad(self, data):

        last_dim = data.shape[-1]
        how_many_layers_needed = self.INPUT_SHAPE[-1] - last_dim

        if how_many_layers_needed <= 0:
            return data

        else:
            zeros_for_expanding = np.zeros((data.shape[0], data.shape[1], how_many_layers_needed,))
            expanded_data = np.concatenate((data, zeros_for_expanding), axis=-1)

        return expanded_data

    def augment(self, imgs, labels):

        aug_imgs = []
        aug_labels = []
        iaa.Sequential()

        for i in range(len(imgs)):
            aug = iaa.Sequential([
                iaa.Affine(rotate=(-90, 90), mode="constant", name="MyAffine"),
                iaa.Sometimes(0.5,
                              iaa.PiecewiseAffine(scale=(0.01, 0.05)))
            ])

            seq_imgs_deterministic = aug.to_deterministic()

            aug_imgs.append(seq_imgs_deterministic.augment_image(imgs[i]))
            aug_labels.append(seq_imgs_deterministic.augment_image(labels[i]))

        return aug_imgs, np.array(aug_labels)

    def _create_one_hot_label(self, label_mask):
        mask = np.zeros((self.batch_size, self.INPUT_SHAPE[0], self.INPUT_SHAPE[1], self.INPUT_SHAPE[2],
                         self.classes))

        for c in range(len(label_mask)):
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