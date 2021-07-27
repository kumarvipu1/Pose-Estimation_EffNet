from __future__ import print_function, division
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import config
from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import numpy as np
import pickle
# Ignore warning
import warnings
warnings.filterwarnings("ignore")


class LoadData(Dataset):
    def __init__(self, annotation_file, root_dir, train=True, transform=None):
        super().__init__()
        with open(annotation_file, 'rb') as handle:
            kp = pickle.load(handle)
        self.data = kp
        self.root_dir = root_dir
        self.category_names = ['left_eye_center_x', 'left_eye_center_y',
                               'right_eye_center_x', 'right_eye_center_y', 'left_eye_inner_corner_x',
                               'left_eye_inner_corner_y', 'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
                               'right_eye_inner_corner_x', 'right_eye_inner_corner_y', 'right_eye_outer_corner_x',
                               'right_eye_outer_corner_y', 'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
                               'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y', 'right_eyebrow_inner_end_x',
                               'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
                               'nose_tip_x', 'nose_tip_y', 'mouth_left_corner_x', 'mouth_left_corner_y',
                               'mouth_right_corner_x', 'mouth_right_corner_y', 'mouth_center_top_lip_x',
                               'mouth_center_top_lip_y', 'mouth_center_bottom_lip_x', 'mouth_center_bottom_lip_y']
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.data['image'])

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.data['image'][idx])
        image = io.imread(img_name)
        kpt = self.data['keypoint'][idx]
        # landmarks = np.array([landmarks])
        kpt = kpt.astype('float')
        sample = {'image': image, 'keypoints': kpt}

        if self.transform:
            sample = self.transform(sample)

        return sample

## Plot 2nd image of the batch

def show_landmarks_batch(sample_batched):
    """Show image with landmarks for a batch of samples."""
    images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['keypoints']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                    landmarks_batch[i, :, 1].numpy(),
                    s=10, marker='.', c='r')

        plt.title('Batch from dataloader')


if __name__ == "__main__":
    annot_file = 'C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/annotation_train.pickle'
    img_path = 'C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/images'
    transformed_dataset = LoadData(annotation_file=annot_file,
                                   root_dir=img_path,
                                   transform=transforms.Compose([
                                       config.Rescale(256),
                                       config.RandomCrop(224),
                                       config.ToTensor()
                                   ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]

        print(i, sample['image'].size(), sample['keypoints'].size())

        if i == 3:
            break

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)


    # Helper function to show a batch
    def show_landmarks_batch(sample_batched):
        """Show image with landmarks for a batch of samples."""
        images_batch, landmarks_batch = \
            sample_batched['image'], sample_batched['keypoints']
        batch_size = len(images_batch)
        im_size = images_batch.size(2)

        grid = utils.make_grid(images_batch)
        plt.imshow(grid.numpy().transpose((1, 2, 0)))

        for i in range(batch_size):
            plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size,
                        landmarks_batch[i, :, 1].numpy(),
                        s=10, marker='.', c='r')

            plt.title('Batch from dataloader')


    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['keypoints'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
    '''
    ds = LoadData(annotation_file='C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/annotation_train.pickle',
                  train=True, transform=config.train_transforms,
                  root_dir = 'C:/Users/44745/Desktop/Facial Keypoint Detection Competition/data/train_data/images')
    loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)

    for i_batch, sample_batched in enumerate(loader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['keypoints'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_landmarks_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break
'''
