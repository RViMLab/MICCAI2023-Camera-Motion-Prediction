import cv2
import numpy as np


class ConsecutiveSequences():
    r"""Iterator over consecutive video sequences.

    Args:
        paths (list of str): Paths to videos
        stride (int): Stride in between frames
        max_seq (int): The number of sequences to return
        seq_len (int): Length of returned sequence
        seq_stride (int): Stride inbetween initial frames of sequences, defaults to seq_len
        transforms (list of callables): Transforming videos
    """
    def __init__(self, paths, stride=1, max_seq=None, seq_len=1, seq_stride=None, transforms=None):
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
        self.frame_counts = np.empty(len(paths))
        for idx, p in enumerate(paths):
            video_capture = cv2.VideoCapture(p)
            self.video_captures.append(video_capture)
            self.frame_counts[idx] = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self):
        return self

    def __next__(self):
        r"""Next method for iterator.

        Returns:
            imgs (np.array): Sequence of shape LxHxWxC
        """
        if self.vid_idx < len(self.video_captures):
            if self.transforms:
                imgs = self._sample(self.video_captures[self.vid_idx], self.frame_idx, self.stride, self.transforms[self.vid_idx])
            else:
                imgs = self._sample(self.video_captures[self.vid_idx], self.frame_idx, self.stride)

            # increment counters
            if self.max_seq is not None:
                if self.seq >= self.max_seq:
                    raise StopIteration
                self.seq += 1
            self.frame_idx += self.seq_stride
            if self.frame_counts[self.vid_idx] - self.stride*self.seq_len < self.frame_idx:
                self.frame_idx = 0
                self.vid_idx += 1
            return imgs
        raise StopIteration

    def __len__(self):
        if not self.max_seq:
            return (self.frame_counts / self.seq_stride + 1).astype(np.int).sum()
        else:
            return self.max_seq

    def _sample(self, capture, frame, stride, transforms=None):
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
