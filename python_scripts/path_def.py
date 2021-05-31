import os, sys

#--------------------------------------------------------------------------------------------
# for general path  
#--------------------------------------------------------------------------------------------

orcvio_root_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', ''))
orcvio_cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../" + "cache/"
orcvio_traj_filename = orcvio_cache_path + "stamped_traj_estimate.txt"

#--------------------------------------------------------------------------------------------
# for EuRoC dataset  
#--------------------------------------------------------------------------------------------

euroc_dataset_path = '/media/erl/disk2/euroc/'
euroc_cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../" + "cache/euroc/"

#--------------------------------------------------------------------------------------------
# for KITTI dataset 
#--------------------------------------------------------------------------------------------

kitti_dataset_path = '/media/erl/disk2/kitti/Kitti_all_data/raw_data/'
kitti_cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../" + "cache/kitti/"
kitti_2d_bbox_filename = orcvio_cache_path + 'object_2d_bbox_info.txt'

#--------------------------------------------------------------------------------------------
# for Unity simulation  
#--------------------------------------------------------------------------------------------

unity_cache_path = os.path.dirname(os.path.realpath(__file__)) + "/../" + "cache/unity/"
