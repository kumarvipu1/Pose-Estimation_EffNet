import torch
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import numpy as np
from skimage import io, transform

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 32
NUM_EPOCHS = 150
NUM_WORKERS = 8
CHECKPOINT_FILE = "b0_new.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# Data augmentation for images

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for keypoints because for images,
        # x and y axes are axis 1 and 0 respectively
        #keypoints = keypoints * [new_w / w, new_h / h]
        keypoints_new = np.reshape(keypoints, (-1, 2))
        keypoints_new = keypoints_new * [new_w / w, new_h / h]
        keypoints = keypoints_new.flatten()

        return {'image': img, 'keypoints': keypoints}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        #keypoints = keypoints - [left, top]
        keypoints_new = np.reshape(keypoints, (-1, 2))
        keypoints_new = keypoints_new - [left, top]
        keypoints = keypoints_new.flatten()

        return {'image': image, 'keypoints': keypoints}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, keypoints = sample['image'], sample['keypoints']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))/255
        image = torch.from_numpy(image).to(torch.float)
        keypoints = torch.from_numpy(keypoints).to(torch.float)
        return {'image': image,
                'keypoints': keypoints}