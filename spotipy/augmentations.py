import numpy as np 
import cv2
from abc import ABC, abstractmethod
from .utils import _filter_shape

class Augmentation(ABC):
    @abstractmethod
    def __call__(self, img, points):
        pass

class FlipRot90(Augmentation):
    @staticmethod
    def _flip(p, n, axis=0):
        p = p.copy() 
        p[:,axis] = n-p[:,axis]-1
        return p 
    @staticmethod
    def _transpose(p):
        p = p.copy() 
        p = p[:,[1,0]]
        return p 

    def __call__(self, img, points):
        flipy, flipx = np.random.randint(0,2,2)==0
        transp = np.random.randint(0,2)==0

        if flipx: 
            img = img[:,::-1]
            points = self._flip(points, img.shape[1], axis=1)

        if flipy:
            img = img[::-1]
            points = self._flip(points, img.shape[0], axis=0)

        if transp:
            img = np.swapaxes(img, 0,1)
            points = self._transpose(points)

        return img, points        

class Rotate(Augmentation):
    def __call__(self, img, points):
        center = tuple(np.array(img.shape[1::-1]) / 2)
        angle = np.random.uniform(0,360)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
        img = cv2.warpAffine(img, M, img.shape[1::-1], flags=cv2.INTER_LINEAR)
        points = points-center 
        points = (M_inv[:2,:2]@points.T).T + center
        points = _filter_shape(points, img.shape[:2])

        return img, points
    
class IntensityScaleShift(Augmentation):
    def __init__(self, scale=(.5,2.), shift=(-1.,.1)):
        super().__init__()
        self.scale = scale
        self.shift = shift

    def __call__(self, img, points):
        scale = np.random.uniform(*self.scale)
        shift = np.random.uniform(*self.shift)
        img = img*scale + shift
        return img, points
class AdditiveNoise(Augmentation):
    def __init__(self, sigma=(0,.1)):
        super().__init__()
        self.sigma = sigma

    def __call__(self, img, points):
        sigma = np.random.uniform(*self.sigma)
        noise = sigma*(np.random.normal(0, 1, img.shape)).astype(np.float32)
        img = img + noise
        return img, points
    
class AugmentationPipeline(object):
    def __init__(self,):
        super().__init__()
        self.augmentations = [] 

    def add(self, augmentation, prob=1.):
        self.augmentations.append((augmentation, prob))

    def __call__(self, img, points):
        for a, p in self.augmentations:
            if np.random.uniform(0,1) < p:
                img, points = a(img, points)
        return img, points
