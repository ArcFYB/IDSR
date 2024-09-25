from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import cv2
from scipy import ndimage
from torchvision import transforms
import numpy as np

def apply_diffusion_and_gradient(image, iterations=10, delta=0.14, kappa=15):

    im = np.array(image)# Convert to numpy array

    # Initial condition
    u = im

    # Center pixel distances
    dx = 1
    dy = 1
    dd = np.sqrt(2)

    # 2D finite difference windows
    windows = [
        np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]]),
        np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]]),
        np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]]),
        np.array([[0, 0, 1], [0, -1, 0], [0, 0, 0]]),
        np.array([[0, 0, 0], [0, -1, 0], [0, 0, 1]]),
        np.array([[0, 0, 0], [0, -1, 0], [1, 0, 0]]),
        np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]]),
    ]

    for r in range(iterations):
        # Approximate gradients
        nabla = [ndimage.filters.convolve(u, w) for w in windows]

        # Approximate diffusion function
        diff = [1. / (1 + (n / kappa) ** 2) for n in nabla]

        # Update image
        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [(1 / (dd ** 2)) * diff[i] * nabla[i] for i in range(4, 8)]
        u = u + delta * (sum(terms))

    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )
    # Apply kernels to the image
    Ix = ndimage.filters.convolve(u, Kx)
    Iy = ndimage.filters.convolve(u, Ky)

    # Return norm of (Ix, Iy)
    G = np.hypot(Ix, Iy)
    
    # Normalize to range [-1, 1]
    u = (u - np.min(u)) / (np.max(u) - np.min(u)) * 2 - 1
    
    u = u.astype(np.float32) * 0.1
    
    trans = transforms.ToTensor()
    u = trans(u)
    
    return u

def EdgeDetection(img_SR, min_max=(0, 1)):
    numpy_image = np.array(img_SR)
    img_SR_canny = cv2.Canny(numpy_image, 100, 200)
    trans = transforms.ToTensor()
    # img_SR_canny = [img * (min_max[1] - min_max[0]) + min_max[0] for img in img_SR_canny]
    img_SR_canny = trans(img_SR_canny)

    return img_SR_canny

class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            self.sr_path = Util.get_paths_from_images(
                '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            self.hr_path = Util.get_paths_from_images(
                '{}/hr_{}'.format(dataroot, r_resolution))
            if self.need_LR:
                self.lr_path = Util.get_paths_from_images(
                    '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:
            img_HR = Image.open(self.hr_path[index]).convert("RGB")
            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.need_LR:
                img_LR = Image.open(self.lr_path[index]).convert("RGB")
        if self.need_LR:
            # img_SR_canny = EdgeDetection(img_SR)
            img_SR_canny = apply_diffusion_and_gradient(img_SR.convert('L'))
            [img_LR, img_SR, img_HR] = Util.transform_augment(
                [img_LR, img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'Index': index, 'ED': img_SR_canny}
        else:
            # img_SR_canny = EdgeDetection(img_SR)
            img_SR_canny = apply_diffusion_and_gradient(img_SR.convert('L'))
            [img_SR, img_HR] = Util.transform_augment(
                [img_SR, img_HR], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'Index': index, 'ED': img_SR_canny+img_HR}
