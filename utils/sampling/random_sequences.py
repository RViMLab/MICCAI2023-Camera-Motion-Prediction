import cv2
import numpy as np
from typing import List, Callable


class RandomSequences():
    r"""Iterator over uniformly sampled random video sequences.

    Args:
        max_seq (int): The number of sequences to return
        paths (list of str): Paths to videos
        seq_len (int): Length of returned sequence
        strides (list of int): Strides in between single frames to randomly sample from
        transforms (list of callables): Transforming videos
        verbose (bool): Return verbose output if true
    """
    def __init__(self, max_seq: int, paths: List[str], seq_len: int=1, strides: List[int]=[1], transforms: Callable=None, verbose: bool=False):
        self.video_captures = []
        self.prob_vid = []
        self.seq = 0
        self.max_seq = max_seq
        self.seq_len = seq_len
        self.strides = strides
        self.transforms = transforms
        self.verbose = verbose
        frame_counts = np.empty(len(paths))
        for idx, p in enumerate(paths):
            video_capture = cv2.VideoCapture(p)
            self.video_captures.append(video_capture)
            frame_counts[idx] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for fc in frame_counts:
            self.prob_vid.append(fc/frame_counts.sum())

    def __del__(self):
        for i in range(len(self.video_captures)):
            self.video_captures[i].release()

    def __iter__(self):
        return self

    def __next__(self):
        r"""Next method for iterator.

        Returns:
            imgs (np.array): Randomly sampled sequence of shape LxHxWxC

            if self.verbose == True:
                vid_idx (int): Video index, as sorted in paths
                frame_idx (int): Initial frame index in video
        """
        if self.seq < self.max_seq:
            vid_idx = np.random.choice(len(self.video_captures), 1, p=self.prob_vid)
            frame_idx = np.random.randint(
                0, 
                self.video_captures[vid_idx[0]].get(cv2.CAP_PROP_FRAME_COUNT) - 1 - max(self.strides)*self.seq_len, 
                1
            )
            stride = np.random.choice(self.strides, 1)[0]
            self.seq += 1
            
            if self.transforms:
                if self.verbose:
                    return self._sample(self.video_captures[vid_idx.item(0)], frame_idx.item(0), stride, self.transforms[vid_idx]), vid_idx, frame_idx
                else:
                    return self._sample(self.video_captures[vid_idx.item(0)], frame_idx.item(0), stride, self.transforms[vid_idx])
            else:
                if self.verbose:
                    return self._sample(self.video_captures[vid_idx.item(0)], frame_idx.item(0), stride), vid_idx, frame_idx
                else:
                    return self._sample(self.video_captures[vid_idx.item(0)], frame_idx.item(0), stride)

        self.__del__()
        raise StopIteration

    def __len__(self):
        return self.max_seq

    def _sample(self, capture: cv2.VideoCapture, frame: int, stride:=1, transforms: Callable=None):
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

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', type=str, nargs='+', required=True, help='Set list of paths to videos.')
    parser.add_argument('-m', '--max_seq', type=int, default=10, help='Set number of frames to return.')
    args = parser.parse_args()

    random_seq = RandomSequences(max_seq=args.max_seq, paths=args.paths)

    for rs in random_seq:

        cv2.imshow('random_frame', rs[0])
        cv2.waitKey()
