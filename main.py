if __name__ == '__main__':
    import argparse
    import cv2
    from tqdm import tqdm

    from utils.sampling import ConsecutiveSequences
    from utils.transforms import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--paths', type=str, nargs='+', required=True, help='Set list of paths to videos.')
    parser.add_argument('-m', '--max_seq', type=int, default=None, help='Set number of frames to return.')
    args = parser.parse_args()

    resize = Resize((1000, 500))
    crop = Crop([0, 0], [200, 200])
    compose = Compose([resize, crop])


    consecutive_seq = ConsecutiveSequences(paths=args.paths, max_seq=args.max_seq, seq_stride=100, seq_len=1, stride=1, transforms=None)

    for cs in tqdm(consecutive_seq):

        for i in range(cs.shape[0]):
            cv2.imshow('random_frame', cs[i])
            cv2.waitKey()
