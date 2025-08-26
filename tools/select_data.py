import numpy as np
import os
import sys
import shutil


wash_data_txt    = '/home/jiyarong/MassSpectrumCls_jia_2_mp/datas/fold4.txt'
wash_data_path   = '/home/data/jiyarong/thyroid_wash_data'


def get_data_dir(data_path):
    with open(data_path, encoding='utf-8') as f:
        data_infos = f.readlines()

    data_paths = []
    for data_info in data_infos:
        data_path, gt = data_info.split(' ')
        data_paths.append(data_path)
        
    return data_paths


if __name__ == "__main__":
    
    data_paths = get_data_dir(wash_data_txt)
    for origin_datapath in data_paths:
        # 获取文件名
        data_name = os.path.basename(origin_datapath)
        # 获取上一级目录名
        class_name = os.path.basename(os.path.dirname(origin_datapath))
        new_save_path = os.path.join(wash_data_path, class_name, data_name)
        # if not os.path.exists(new_save_path):
        #     os.makedirs(new_save_path)
        # 使用 shutil.copy() 复制图像文件
        shutil.copy(origin_datapath, new_save_path)
