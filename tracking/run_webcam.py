import _init_paths
import os
import cv2
import torch
import torch.utils.data
import random
import argparse
import numpy as np

import lib.models.models as models

from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from lib.eval_toolkit.pysot.datasets import VOTDataset
from lib.eval_toolkit.pysot.evaluation import EAOBenchmark
from lib.tracker.lighttrack import Lighttrack


def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--arch', dest='arch', help='backbone architecture')
    parser.add_argument('--resume', type=str, help='pretrained model')
    parser.add_argument('--dataset', default='VOT2019', help='dataset test')
    parser.add_argument('--epoch_test', default=False, type=bool, help='multi-gpu epoch test flag')
    parser.add_argument('--video', default=None, type=str, help='test a video in benchmark')
    parser.add_argument('--stride', type=int, help='network stride')
    parser.add_argument('--even', type=int, default=0)
    parser.add_argument('--path_name', type=str, default='NULL')
    args = parser.parse_args()

    return args


DATALOADER_NUM_WORKER = 1


def collate_fn(x):
    return x[0]


def track(siam_tracker, siam_net, args):
    start_frame, toc = 0, 0
    snapshot_dir = os.path.dirname(args.resume)
    result_dir = os.path.join(snapshot_dir, '../..', 'result')
    model_name = snapshot_dir.split('/')[-1]

    # save result to evaluate
    tracker_path = os.path.join(result_dir, args.dataset, model_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    cap = cv2.VideoCapture("/media/pxierra/4ddb33c4-42d9-4544-b7b4-796994f061ce/data/baby_video/video/01.mp4")

    with torch.no_grad():
        count = 0
        while(True):
            ret, frame = cap.read()
            tic = cv2.getTickCount()
            if count == 0:  # init
                cv2.putText(frame, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                           1.5, (0, 0, 0), 1)

                x, y, w, h = cv2.selectROI("demo", frame, fromCenter=False)
                frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target_pos = np.array([x+w/2, y+h/2])
                target_sz = np.array([w, h])
                state = siam_tracker.init(frame_disp, target_pos, target_sz, siam_net)
                count = count + 1

            else:  # tracking
                frame_disp = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                state = siam_tracker.track(state, frame_disp)

                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                cv2.rectangle(frame, (int(location[0]), int(location[1])), (int(location[0]+location[2]), int(location[1]+location[3])), (0, 0, 255))

                cv2.imshow("img", frame)
                cv2.waitKey(33)

            toc += cv2.getTickCount() - tic

    # print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = args.dataset
    info.epoch_test = args.epoch_test
    info.stride = args.stride

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = args.dataset
    siam_info.epoch_test = args.epoch_test
    siam_info.stride = args.stride
    # build tracker
    siam_tracker = Lighttrack(siam_info, even=args.even)
    # build siamese network
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)

    print('===> init Siamese <====')

    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()

    track(siam_tracker, siam_net, args)


if __name__ == '__main__':
    main()
