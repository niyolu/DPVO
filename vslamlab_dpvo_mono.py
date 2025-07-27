import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.stream import image_stream
from dpvo.utils import Timer

# if __name__ == "__main__":
#     import debugpy
#     debugpy.listen(("localhost", 5678))
#     print("Waiting for debugger attach on port 5678...")
#     debugpy.wait_for_client()

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)
    

@torch.no_grad()
def run(cfg, network, sequence_path, rgb_txt, calibration_yaml, viz=False, timeit=False):

    slam = None
    queue = Queue(maxsize=8)

    reader = Process(target=image_stream, args=(queue, sequence_path, rgb_txt, calibration_yaml))
    reader.start()

    while 1:
        print("trying to get from queue")
        (t, image, intrinsics) = queue.get()
        print("got data from queue: ", t)
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if slam is None:
            _, H, W = image.shape
            slam = DPVO(cfg, network, ht=H, wd=W, viz=viz)

        with Timer("SLAM", enabled=timeit):
            print("running dpvo t=", t)
            slam(t, image, intrinsics)
        # slam(t, image, intrinsics)

    reader.join()

    points = slam.pg.points_.cpu().numpy()[:slam.m]
    colors = slam.pg.colors_.view(-1, 3).cpu().numpy()[:slam.m]

    return slam.terminate(), (points, colors, (*intrinsics, H, W))

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sequence_path", type=str, help="path to image directory")
    parser.add_argument("--calibration_yaml", type=str, help="path to calibration file")
    parser.add_argument("--rgb_txt", type=str, help="path to image list")
    parser.add_argument("--exp_folder", type=str, help="path to save results")
    parser.add_argument("--exp_it", type=str, help="experiment iteration")
    parser.add_argument("--settings_yaml", type=str, help="settings_yaml")
    parser.add_argument("--verbose", type=str, help="verbose")

    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--name', type=str, help='name your run', default='result')
    parser.add_argument('--timeit', action='store_true')
    parser.add_argument('--opts', nargs='+', default=[])

    args, _ = parser.parse_known_args()

    cfg.merge_from_file(args.settings_yaml)
    cfg.merge_from_list(args.opts)

    print("Running with config...")
    print("SKR")
    print(args.settings_yaml)

    (poses, tstamps), (points, colors, calib) = run(cfg, args.network, 
                                                    args.sequence_path, args.rgb_txt, args.calibration_yaml, 
                                                    bool(int(args.verbose)), args.timeit)
    trajectory = PoseTrajectory3D(positions_xyz=poses[:,:3], orientations_quat_wxyz=poses[:, [6, 3, 4, 5]], timestamps=tstamps)

    keyFrameTrajectory_txt = os.path.join(args.exp_folder, args.exp_it.zfill(5) + '_KeyFrameTrajectory' + '.txt')
    file_interface.write_tum_trajectory_file(keyFrameTrajectory_txt, trajectory)
 
if __name__ == '__main__':
    main()
