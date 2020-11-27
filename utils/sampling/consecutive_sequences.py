import cv2
import numpy as np
from typing import List, Callable


class ConsecutiveSequences():
    r"""Iterator over consecutive video sequences.

    Args:
        paths (list of str): Paths to videos
        stride (int): Stride in between frames
        max_seq (int): The number of sequences to return
        seq_len (int): Length of returned sequence
        seq_stride (int): Stride inbetween initial frames of sequences, defaults to seq_len
        transforms (list of callables): Transforming videos
        verbose (bool): Return verbose output if true
    """
    def __init__(self, paths: List[str], stride: int=1, max_seq: int=None, seq_len: int=1, seq_stride: int=None, transforms: Callable=None, verbose: bool=False):
        self.video_captures = []
        self.seq = 0
        self.max_seq = max_seq
        self.seq_len = seq_len
        self.vid_idx = 0
        self.frame_idx = 0
        self.stride = stride
        self.seq_stride = seq_stride
        if not seq_stride:
            self.seq_stride = seq_len
        self.transforms = transforms
        self.verbose = verbose
        self.frame_counts = np.empty(len(paths))
        for idx, p in enumerate(paths):
            video_capture = cv2.VideoCapture(p)
            self.video_captures.append(video_capture)
            self.frame_counts[idx] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __del__(self):
        for i in range(len(self.video_captures)):
            self.video_captures[i].release()

    def __iter__(self):
        return self

    def __next__(self):
        r"""Next method for iterator.

        Returns:
            imgs (np.array): Sequence of shape LxHxWxC

            if self.verbose == True:
                vid_idx (int): Video index, as sorted in paths
                frame_idx (int): Initial frame index in video
        """
        if self.vid_idx < len(self.video_captures):
            if self.transforms:
                imgs = self._sample(self.video_captures[self.vid_idx], self.frame_idx, self.stride, self.transforms[self.vid_idx])
            else:
                imgs = self._sample(self.video_captures[self.vid_idx], self.frame_idx, self.stride)

            vid_idx = self.vid_idx
            frame_idx = self.frame_idx

            # increment counters
            if self.max_seq is not None:
                if self.seq >= self.max_seq:
                    self.__del__()
                    raise StopIteration
                self.seq += 1
            self.frame_idx += self.seq_stride
            if self.frame_counts[self.vid_idx] - self.stride*self.seq_len < self.frame_idx:
                self.video_captures[self.vid_idx].release()
                self.frame_idx = 0
                self.vid_idx += 1
            if self.verbose:
                return imgs, vid_idx, frame_idx
            else:
                return imgs

        self.__del__()
        raise StopIteration

    def __len__(self):
        if not self.max_seq:
            return (self.frame_counts / self.seq_stride + 1).astype(np.int).sum()
        else:
            return self.max_seq

    def _sample(self, capture: cv2.VideoCapture, frame: int, stride: int, transforms: Callable=None):
        r"""Return a sequence of images from a videos capture.

        Args:
            capture (cv2.VideoCapure): Video capture
            frame (int): Frame to return from capture
            stride (int): Stride between frames
            transforms (callable): Transforms to be applied

        Return:
            imgs (np.array): Frames to return of shape LxHxWxC
        """
        imgs = []
        for i in range(self.seq_len):
            capture.set(1, frame+i*stride) # https://stackoverflow.com/questions/33650974/opencv-python-read-specific-frame-using-videocapture
            _, img = capture.read()
            if transforms:
                img = transforms(img)
            imgs.append(img)
        return np.array(imgs)


if __name__ == '__main__':
    import argparse
    import cv2
    from tqdm import tqdm

    from utils.transforms import Resize

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', type=str, nargs='+', required=True, help='Set list of paths to videos.')
    parser.add_argument('-m', '--max_seq', type=int, default=None, help='Set number of frames to return.')
    args = parser.parse_args()

    consecutive_seq = ConsecutiveSequences(paths=args.paths, max_seq=args.max_seq, seq_stride=2000, seq_len=1, stride=1)

    for cs in tqdm(consecutive_seq):

        for i in range(cs.shape[0]):
            cv2.imshow('random_frame', cs[i])
            cv2.waitKey()
