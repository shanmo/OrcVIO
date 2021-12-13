## OrcVIO 

This repo implements [OrcVIO: Object residual constrained Visual-Inertial Odometry](https://moshan.cf/orcvio_githubpage/)

## Dependencies 

The core algorithm depends on `Eigen`, `Boost`, `Suitesparse`, `Ceres`, `OpenCV`, `Sophus`. 

## Usage 

### Non-ROS version 
```
cd ORCVIO
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=Release ..
make
```
and then 
``` shellsession
cd YOUR_PATH/ORCVIO/
./build/orcvio ~/YOUR_PATH_TO_DATASET/MH_01/mav0/imu0/data.csv ~/YOUR_PATH_TO_DATASET/MH_01/mav0/cam0/data.csv ~/YOUR_PATH_TO_DATASET/MH_01/mav0/cam0/data ~/YOUR_PATH/LARVIO/config/euroc.yaml
```
eg 
``` shellsession
cd YOUR_PATH/ORCVIO/
./build/orcvio /media/erl/disk2/euroc/MH_01_easy/mav0/imu0/data.csv /media/erl/disk2/euroc/MH_01_easy/mav0/cam0/data.csv /media/erl/disk2/euroc/MH_01_easy/mav0/cam0/data /home/erl/Workspace/orcvio_cpp/config/euroc.yaml
```
to debug using VS code, define `VS_DEBUG_MODE`, then run
```
cd build
cmake -DCMAKE_EXPORT_COMPILE_COMMANDS=YES ..
make
```
this will generate the `compile_commands.json` in the `build` folder 

### ROS version 
- how to build 
```
cd YOUR_PATH/ORCVIO/ros_wrapper
catkin_make
```
and then 
```
source YOUR_PATH/ORCVIO/ros_wrapper/devel/setup.bash
roslaunch orcvio orcvio_euroc.launch
```
- how to setup front end refer to wiki 
- we also have an OrcVIO-lite version which only uses bounding boxes, no keypoints, this mode can be enabled by changing the flag in the launch file 
- visualization 
> ![demo](assets/demo-unity.gif)

## Evaluation 

### unit test (Non-ROS version)

The tests are compiled by default and have to be run from top of the project directory

``` shellsession
$ mkdir -p build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Debug -DBUILD_TESTING=True ..
$ make
$ cd .. && ./build/orcvio_tests ; cd -
```

Or directly run `make test` from `build` directory and the test logs are saved in `build/Testing/Temporary/LastTest.log`.

### VIO 

1. evaluation on EuRoC dataset 
- cd into `python_scripts/trajectory_eval/` folder and `python batch_run_euroc.py`
- [comparison with LARVIO](https://github.com/moshanATucsd/orcvio_cpp/blob/master/eval_results/orcvio_vs_larvio/orcvio_vs_larvio_euroc.md) 
- [comparison of left and right purbation in VIO](https://github.com/moshanATucsd/orcvio_cpp/blob/master/eval_results/left_vs_right_perturb/orcvio_result.md) 

2. trajectory evaluation using KITTI dataset error metric
- cd into `python_scripts/trajectory_eval/` folder and `python traj_eval.py`

### Object mapping 

- object map result is saved under `cache/object_map`
- cd into `python_scripts/object_map_eval/` folder and `python unity_object_iou_eval.py` to get the 3D IOU results when using Unity simulation; to enable debug mode use `python3 unity_object_iou_eval.py` 
- cd into `python_scripts/object_map_eval/` folder and `python3 kitti_object_iou_eval_single_sequence.py` to get the 3D IOU results for one sequence in KITTI, `python kitti_construct_pr_table_all_sequences.py` to get the PR table for KITTI

## Citation 

```
@inproceedings{orcvio,
  title = {OrcVIO: Object residual constrained Visual-Inertial Odometry},
  author={M. {Shan} and Q. {Feng} and N. {Atanasov}},
  year = {2020},
  booktitle={IEEE Intl. Conf. on Intelligent Robots and Systems (IROS).},
  url = {https://moshanatucsd.github.io/orcvio_githubpage/},
  pdf = {https://arxiv.org/abs/2007.15107}
}
```

## License

```
MIT License
Copyright (c) 2020 ERL at UCSD
```

## References 

- [LARVIO](https://github.com/PetWorm/LARVIO)
- [MSCKF](https://github.com/KumarRobotics/msckf_vio)
- [OpenVINS](https://github.com/rpng/open_vins)
- [OpenVINS evaluation](https://github.com/symao/open_vins)
