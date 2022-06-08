import argparse
import importlib
import os
import random
import shutil
import cv2

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
import scipy.spatial
from nerf.atlantic_datasets.block_selector import selectors
from nerf.atlantic_datasets.block_selector import n_images
from nerf.atlantic_datasets.block_selector import poses

def merge_config_file(config, config_path, allow_invalid=False):
    """
    Load yaml config file if specified and merge the arguments
    """
    if config_path is not None:
        with open(config_path, "r") as config_file:
            new_config = yaml.safe_load(config_file)
        invalid_args = list(set(new_config.keys()) - set(config.keys()))
        if invalid_args and not allow_invalid:
            raise ValueError(f"Invalid args {invalid_args} in {config_path}.")
        config.update(new_config)

## calculates weight
## weight = sum_i((cosine(yaw)/(1+distance)))
def get_weight(reference_id,target_block,img_poses):
    reference_pose = img_poses[reference_id]
    weight_sum=0
    for target_id in target_block:
        target_pose = img_poses[target_id]
        dis = np.linalg.norm(target_pose[:3]-reference_pose[:3],2)
        target_R=scipy.spatial.transform.Rotation.from_quat(target_pose[[4, 5, 6, 3]]).as_matrix()
        reference_R=scipy.spatial.transform.Rotation.from_quat(reference_pose[[4, 5, 6, 3]]).as_matrix()
        delta_R=target_R..transpose()*reference_R
        cosine_yaw=(delta_R.trace()-1)/2
        weight_sum += cosine_yaw/(1+dis)
    return weight_sum
        
 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str)
    parser.add_argument("--out", "-o", type=str, default="")
    args = parser.parse_args()

    default_config_path = "./configs/default.yaml"
    with open(default_config_path, "r") as config_file:
        opt = yaml.safe_load(config_file)
    opt["config"] = args.config if args.config else default_config_path
    opt["out"] = args.out
    merge_config_file(opt, args.config, allow_invalid=True)
    opt = OmegaConf.create(opt)
    print(opt)

    config_name = os.path.splitext(os.path.basename(opt.config))[0]
    workspace = os.path.join("logs", config_name, opt.module)
    
    save_path_root = os.path.join(workspace, "final_results")
    os.makedirs(save_path_root, exist_ok=True)
    for i in range(n_images):
        img_path_list=[]
        merge_weights=[]
        weight_sum=0
        merge_img=None
        for j in range(len(selectors)):
            name='kitti-sub'+format(str(j), '0>2s')
            if(i in selectors[name]['test']):
                index_i = selectors[name]['test'].index(i)
                img_path=os.path.join(workspace, "results", f"{name}_{index_i:04d}.png")
                img_path_list.append(img_path)
                #calculate merge weight
                weight = get_weight(i,selectors[name]['test'],poses)
                merge_weights.append(weight)
                img = cv2.imread(img_path).astype("float32")
                
                weight_sum += weight
                if(merge_img is None):
                    merge_img = img
                else:
                    merge_img += weight * img
        #merge image
        merge_img = (merge_img / weight_sum).astype("uint8")   
        save_path = os.path.join(save_path_root, f"{i:04d}.png")
        cv2.imwrite(save_path, merge_img)
    
    # Video
    video_path = os.path.join(workspace, "video.webm")
    ffmpeg_bin = "ffmpeg"
    frame_regexp = os.path.join(workspace, "final_results", "%04d.png")
    pix_fmt = "yuva420p"
    ffmcmd = (
        '%s -r %d -i %s -vf pad="width=ceil(iw/2)*2:height=ceil(ih/2)*2" -c:v libvpx-vp9 -crf %d -b:v 0 -pix_fmt %s -y -an %s'
        % (ffmpeg_bin, opt.test.fps, frame_regexp, opt.test.crf, pix_fmt, video_path)
    )
    ret = os.system(ffmcmd)
    if ret != 0:
        raise RuntimeError("ffmpeg failed!")

    # Output
    if opt.out != "":
        if not opt.out.startswith("s3://"):
            if os.path.isdir(opt.out):
                opt.out = os.path.join(opt.out, f"{config_name}.webm")
            shutil.copyfile(video_path, opt.out)
        else:
            if not opt.out.endswith(".webm"):
                opt.out = os.path.join(opt.out, f"{config_name}.webm")
            osscmd = f"aws --endpoint-url=http://oss.hh-b.brainpp.cn s3 cp {video_path} {opt.out}"
            ret = os.system(osscmd)
            if ret != 0:
                raise RuntimeError("oss cp failed!")
            print(f"Video path: http://oss.iap.hh-b.brainpp.cn/{opt.out[5:]}")
