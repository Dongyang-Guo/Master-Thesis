# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
from argparse import ArgumentParser
import os
import time
from mmdet.apis import (async_inference_detector, inference_detector,
                        init_detector, show_result_pyplot)
def main(img_list,out_list,config_path,cpt,device,score_thr,palette):
    # build the model from a config file and a checkpoint file

    model = init_detector(config_path, cpt, device=device)
    path_zip = zip(img_list,out_list)
    for zip_path in list(path_zip):
        img_path,out_file = zip_path
        result = inference_detector(model,img_path)

        show_result_pyplot(model,img_path,result,palette=palette,score_thr=score_thr,out_file=out_file)


if __name__ == '__main__':
    device = 'cuda:0'
    score_thr = 0.5
    file_path = r'demo\\'
    config_path = 'work_dirs\solov2_light_x101_dcn++\solov2_light_x101_dcn++.py'
    cpt = r'work_dirs\solov2_light_x101_dcn++\epoch_99.pth'
    result_save_path = file_path + str(time.strftime(('%Y-%m-%d-%H-%M'),time.localtime())) + '_' +str(score_thr).replace('.','')+'_result'
    palette = 'coco'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    img_list = []
    out_list = []
    for img in os.listdir(file_path):
        if img.endswith('jpg'):
            img_path = os.path.join(file_path,img)
            out_file = result_save_path + '/' +img.replace('.jpg','') + '_result.png'
            img_list.append(img_path)
            out_list.append(out_file)
    main(img_list,out_list,config_path,cpt,device,score_thr,palette)
