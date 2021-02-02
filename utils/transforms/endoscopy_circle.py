import numpy as np


class EndoscopyCircle(object):
    def __call__(self, img: np.array, center: tuple=None, radius: float=None):
        return self._circularNoisyMask(img, center, radius)

    def _circularNoisyMask(self, img: np.array, center: tuple=None, radius: float=None) -> np.array:
        """Puts a circular noisy mask on top of an image, mask from https://stackoverflow.com/questions/17394882/how-can-i-add-new-dimensions-to-a-numpy-array.

        Args:
            img (np.array): uint8 image to be masked of shape HxWxC
            center (tuple): Circular mask's center
            radius (float): Circular mask's radius

        Return:
            img (np.array): Masked image
        """
        h, w = img.shape[:-1]

        if center is None: # use the middle of the image
            center = (int(w/2), int(h/2))
        if radius is None: # use the smallest distance between the center and image walls
            radius = min(center[0], center[1], w-center[0], h-center[1])

        Y, X = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

        mask = np.expand_dims(np.where(dist_from_center <= radius, 1, 0), axis=-1).astype(np.uint8)
        img = mask*img + ((1-mask)*np.random.random(img.shape)*255).astype(np.uint8)
        return img

if __name__ == '__main__':
    import cv2

    ec = EndoscopyCircle()

    img = np.load('utils/transforms/sample.npy')

    # sample 10 loops
    for _ in range(10):
        offset = (np.random.rand(2)*2 - 1)*img.shape[:-1]/2
        center = (img.shape[1]/2 + offset[1], img.shape[0]/2 + offset[0])
        radius = np.random.random(1)*img.shape[0]/2.

        masked_img = ec(img, center, radius)

        cv2.imshow('masked_img', masked_img)
        cv2.waitKey()
