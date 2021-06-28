import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', type=str, required=True, help="path to h5 result file")
parser.add_argument('-d', '--frm-dir', type=str, required=True, help="path to frame directory")
#parser.add_argument('-i', '--idx', type=str, default="video_11", help="which key to choose")
parser.add_argument('--fps', type=int, default=30, help="frames per second")
parser.add_argument('--width', type=int, default=320, help="frame width")
parser.add_argument('--height', type=int, default=180, help="frame height")
parser.add_argument('--save-dir', type=str, default='log', help="directory to save")
#parser.add_argument('--save-name', type=str, default='summary.mp4', help="video name to save (ends with .mp4)")
args = parser.parse_args()

def frm2video(frm_dir, summary, vid_writer):
    for idx, val in enumerate(summary):
        if val == 1:
            # here frame name starts with '000001.jpg'
            # change according to your need
            frm_name = str(idx+1).zfill(6) + '.jpg'
            frm_path = osp.join(frm_dir, frm_name)
            frm = cv2.imread(frm_path)
            #frm = cv2.resize(frm, (width, height))
            vid_writer.write(frm)

if __name__ == '__main__':
    width = 1280
    height = 720
    h5_res = h5py.File(args.path, 'r')
    keys = h5_res.keys()
    n=0
    for key in keys:
        print(key)
        n=n+1
        if not osp.exists(args.save_dir):
            os.mkdir(args.save_dir)
        vid_writer = cv2.VideoWriter(
            osp.join(args.save_dir, "summaryFortennis" + str(n)+".mp4"),
            cv2.VideoWriter_fourcc(*'mp4v'),
            args.fps,
            (width, height),
        )
        summary = h5_res[key]['machine_summary'][...]
        
        frm2video(args.frm_dir, summary, vid_writer)
        vid_writer.release()
    h5_res.close()