import numpy as np 
import os, sys, glob  

import kitti_sem_data_loader
import kitti_mapping_eval 

# kitti_date = '2011_09_26'
# kitti_drive = '0022'
# start_index = 0
# end_index = 800

# kitti_date = '2011_09_26'
# kitti_drive = '0023'
# start_index = 0
# end_index = 470

kitti_date = '2011_09_26'
kitti_drive = '0095'
start_index = 0
end_index = 265

# refer to https://github.com/moshanATucsd/orcvio_cpp/blob/master/eval_results/kitti_eval/eval_info.md
# for which method to choose for each sequence 
# object_label_status = 'tracklet_label'
object_label_status = 'detection_label'

DL = kitti_sem_data_loader.KittiSemDataLoader(
    kitti_date, kitti_drive, end_index, object_label_status)

OE = kitti_mapping_eval.ObjectEvaluator(DL, start_index, end_index)

RL = kitti_mapping_eval.ResultsLogger(DL.pr_table_dir, OE)
