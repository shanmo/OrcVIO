/*
 * COPYRIGHT AND PERMISSION NOTICE
 * Penn Software MSCKF_VIO
 * Copyright (C) 2017 The Trustees of the University of Pennsylvania
 * All rights reserved.
 */

// The original file belongs to MSCKF_VIO (https://github.com/KumarRobotics/msckf_vio/)
// Tremendous changes have been made to use it in orcvio

#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

#include <Eigen/SVD>
#include <Eigen/QR>
#include <Eigen/SparseCore>
#include <Eigen/SPQRSupport>
#include <boost/math/distributions/chi_squared.hpp>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>

#include <orcvio/orcvio.h>
#include <orcvio/utils/math_utils.hpp>
#include <orcvio/utils/se3_ops.hpp>

#include <opencv2/core/utility.hpp>

using namespace std;
using namespace Eigen;

namespace orcvio {

// Static member variables in IMUState class.
StateIDType IMUState::next_id = 0;
Vector3d IMUState::gravity = Vector3d(0, 0, -GRAVITY_ACCELERATION);
Isometry3d IMUState::T_imu_body = Isometry3d::Identity();

// Static member variables in Feature class.
FeatureIDType Feature::next_id = 0;
Feature::OptimizationConfig Feature::optimization_config;


OrcVIO::OrcVIO(std::string& config_file_):
    is_gravity_set(false),
    is_first_img(true),
    config_file(config_file_) {
  return;
}


OrcVIO::~OrcVIO() {
  
  fImuState.close();
  fTakeOffStamp.close();

  return;
}


bool OrcVIO::loadParameters() {
  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
      cout << "config_file error: cannot open " << config_file << endl;
      return false;
  }

  use_left_perturbation_flag = fsSettings["use_left_perturbation_flag"];
  use_closed_form_cov_prop_flag = fsSettings["use_closed_form_cov_prop_flag"];
  use_larvio_flag = fsSettings["use_larvio_flag"];
  discard_large_update_flag = fsSettings["discard_large_update_flag"];

  features_rate = fsSettings["pub_frequency"];
  imu_rate = fsSettings["imu_rate"];
  imu_img_timeTh = 1/(2*imu_rate);

  position_std_threshold = fsSettings["position_std_threshold"];
  rotation_threshold = fsSettings["rotation_threshold"];
  translation_threshold = fsSettings["translation_threshold"];
  tracking_rate_threshold = fsSettings["tracking_rate_threshold"];

  max_track_len = fsSettings["max_track_len"];

  // Feature optimization parameters
  Feature::optimization_config.translation_threshold = fsSettings["feature_translation_threshold"];
  Feature::optimization_config.cost_threshold = fsSettings["feature_cost_threshold"];
  Feature::optimization_config.init_final_dist_threshold = fsSettings["init_final_dist_threshold"];

  // Time threshold for resetting FEJ, in seconds, 
  reset_fej_threshold = fsSettings["reset_fej_threshold"];

  // Timestamp synchronization
  td_input = fsSettings["td"];
  state_server.td = td_input;
  estimate_td = (static_cast<int>(fsSettings["estimate_td"]) ? true : false);

  // If estimate extrinsic
  estimate_extrin = (static_cast<int>(fsSettings["estimate_extrin"]) ? true : false);

  // Noise related parameters
  imu_gyro_noise = fsSettings["noise_gyro"];
  imu_acc_noise = fsSettings["noise_acc"];
  imu_gyro_bias_noise = fsSettings["noise_gyro_bias"];
  imu_acc_bias_noise = fsSettings["noise_acc_bias"];
  feature_observation_noise = fsSettings["noise_feature"];  

  // Use variance instead of standard deviation.
  imu_gyro_noise *= imu_gyro_noise;
  imu_acc_noise *= imu_acc_noise;
  imu_gyro_bias_noise *= imu_gyro_bias_noise;
  imu_acc_bias_noise *= imu_acc_bias_noise;
  feature_observation_noise *= feature_observation_noise;

  // Noise of ZUPT measurement
  zupt_noise_v = fsSettings["zupt_noise_v"];
  zupt_noise_p = fsSettings["zupt_noise_p"];
  zupt_noise_q = fsSettings["zupt_noise_q"];
  zupt_noise_v *= zupt_noise_v;
  zupt_noise_p *= zupt_noise_p;
  zupt_noise_q *= zupt_noise_q;

  initial_use_gt = (static_cast<int>(fsSettings["initial_use_gt"]) ? true : false);
  if (initial_use_gt) {

    // very awkward...
    cv::Mat gyro_bias_mat, acc_bias_mat, position_mat, velocity_mat, orientation_mat;
    initial_state_time = fsSettings["initial_state_time"];
    fsSettings["initial_bg"] >> gyro_bias_mat;
    fsSettings["initial_ba"] >> acc_bias_mat;
    fsSettings["initial_pos"] >> position_mat;
    fsSettings["initial_vel"] >> velocity_mat;
    fsSettings["initial_quat"] >> orientation_mat;

    cv::Vec3d gyro_bias_cv = gyro_bias_mat(cv::Rect(0,0,1,3));
    cv::Vec3d acc_bias_cv = acc_bias_mat(cv::Rect(0,0,1,3));
    cv::Vec3d position_cv = position_mat(cv::Rect(0,0,1,3));
    cv::Vec3d velocity_cv = velocity_mat(cv::Rect(0,0,1,3));
    cv::Vec4d orientation_cv = orientation_mat(cv::Rect(0,0,1,4));

    cv::cv2eigen(gyro_bias_cv, init_gyro_bias);
    cv::cv2eigen(acc_bias_cv, init_acc_bias);
    cv::cv2eigen(position_cv, init_position);
    cv::cv2eigen(velocity_cv, init_velocity);
    cv::cv2eigen(orientation_cv, init_orientation);
  }

  prediction_only_flag = (static_cast<int>(fsSettings["prediction_only_flag"]) ? true : false);

  // The initial covariance of orientation and position can be
  // set to 0. But for velocity, bias and extrinsic parameters,
  // there should be nontrivial uncertainty.
  double orientation_cov, position_cov, velocity_cov, gyro_bias_cov, acc_bias_cov;
  orientation_cov = fsSettings["initial_covariance_orientation"];
  position_cov = fsSettings["initial_covariance_position"];
  velocity_cov = fsSettings["initial_covariance_velocity"];
  gyro_bias_cov = fsSettings["initial_covariance_gyro_bias"];
  acc_bias_cov = fsSettings["initial_covariance_acc_bias"];

  double extrinsic_rotation_cov, extrinsic_translation_cov;
  extrinsic_rotation_cov = fsSettings["initial_covariance_extrin_rot"];   
  extrinsic_translation_cov = fsSettings["initial_covariance_extrin_trans"];

  // Imu instrinsics
  calib_imu = (static_cast<int>(fsSettings["calib_imu_instrinsic"]) ? true : false);
  // TODO: read values from config files
  state_server.Ma = Matrix3d::Identity();
  state_server.Tg = Matrix3d::Identity();
  state_server.As = Matrix3d::Zero();
  state_server.M1 << state_server.Ma(1,0),
      state_server.Ma(2,0),
      state_server.Ma(2,1);
  state_server.M2 << state_server.Ma(0,0),
      state_server.Ma(1,1),
      state_server.Ma(2,2);
  state_server.T1 << state_server.Tg(1,0),
      state_server.Tg(2,0),
      state_server.Tg(2,1);
  state_server.T2 << state_server.Tg(0,0),
      state_server.Tg(1,1),
      state_server.Tg(2,2);
  state_server.T3 << state_server.Tg(0,1),
      state_server.Tg(0,2),
      state_server.Tg(1,2);
  state_server.A1 << state_server.As(1,0),
      state_server.As(2,0),
      state_server.As(2,1);
  state_server.A2 << state_server.As(0,0),
      state_server.As(1,1),
      state_server.As(2,2);
  state_server.A3 << state_server.As(0,1),
      state_server.As(0,2),
      state_server.As(1,2);

  // Calculate the dimension of legacy error state
  if (calib_imu)
      LEG_DIM = 46;
  else
      LEG_DIM = 22;

  state_server.state_cov = MatrixXd::Zero(LEG_DIM, LEG_DIM);
  for (int i = 0; i < 3; ++i)
      state_server.state_cov(i, i) = orientation_cov;
  for (int i = 3; i < 6; ++i)
      state_server.state_cov(i, i) = velocity_cov;
  for (int i = 6; i < 9; ++i)
      state_server.state_cov(i, i) = position_cov;
  for (int i = 9; i < 12; ++i)
      state_server.state_cov(i, i) = gyro_bias_cov;
  for (int i = 12; i < 15; ++i)
      state_server.state_cov(i, i) = acc_bias_cov;

  if (estimate_extrin) {
      for (int i = 15; i < 18; ++i)
          state_server.state_cov(i, i) = extrinsic_rotation_cov;
      for (int i = 18; i < 21; ++i)
          state_server.state_cov(i, i) = extrinsic_translation_cov;
  }
  if (estimate_td) {
      state_server.state_cov(21, 21) = 4e-6;
  }
  if (calib_imu) {
      state_server.state_cov.block<24,24>(22,22) =
          1e-4*MatrixXd::Identity(24, 24);
  }

  // Transformation offsets between the frames involved.
  // Note, T_cam_imu is defined in Kalibr, as from imu to camera frame 
  // In Larvio, T_imu_cam is from imu to camera frame 
  // T_cam_imu is different from the one used in MSCKF or OpenVINS, which 
  // is from camera frame to imu 
  cv::Mat T_imu_cam;
  fsSettings["T_cam_imu"] >> T_imu_cam;
  cv::Matx33d R_imu_cam(T_imu_cam(cv::Rect(0,0,3,3)));
  cv::Vec3d t_imu_cam = T_imu_cam(cv::Rect(3,0,1,3));
  Matrix3d R_imu_cam_eigen;
  Vector3d t_imu_cam_eigen;
  cv::cv2eigen(R_imu_cam, R_imu_cam_eigen);
  cv::cv2eigen(t_imu_cam, t_imu_cam_eigen);
  Isometry3d T_imu_cam0;
  T_imu_cam0.linear() = R_imu_cam_eigen;
  T_imu_cam0.translation() = t_imu_cam_eigen;
  Isometry3d T_cam0_imu = T_imu_cam0.inverse();

  state_server.imu_state.R_imu_cam0 = T_cam0_imu.linear().transpose();
  state_server.imu_state.t_cam0_imu = T_cam0_imu.translation() + 0.0*Vector3d::Random();

  // Maximum number of camera states to be stored
  sw_size = fsSettings["sw_size"];

  // If applicate FEJ
  if_FEJ_config = (static_cast<int>(fsSettings["if_FEJ"]) ? true : false);

  // Least observation number for a valid feature
  least_Obs_Num = fsSettings["least_observation_number"];

  // Maximum feature distance for zero velocity detection
  if_ZUPT_valid = (static_cast<int>(fsSettings["if_ZUPT_valid"]) ? true : false);
  if_use_feature_zupt_flag = (static_cast<int>(fsSettings["if_use_feature_zupt_flag"]) ? true : false);
  // Maximum feature distance for zero velocity detection
  zupt_max_feature_dis = fsSettings["zupt_max_feature_dis"];

  use_object_residual_update_cam_pose_flag = fsSettings["use_object_residual_update_cam_pose_flag"];

  // Output files directory
  fsSettings["output_dir"] >> output_dir;

  // Static scene duration, in seconds
  Static_Duration = fsSettings["static_duration"];
  Static_Num = (int)(Static_Duration*features_rate);

  // Grid distribution parameters
  grid_rows = fsSettings["aug_grid_rows"];
  grid_cols = fsSettings["aug_grid_cols"];

  // Maximum number of features in state
  // a positive value is hybrid msckf 
  max_features = fsSettings["max_features_in_one_grid"];
  if (max_features<0) max_features=0;

  // Resolution of camera
  cam_resolution.resize(2);
  cam_resolution[0] = fsSettings["resolution_width"];
  cam_resolution[1] = fsSettings["resolution_height"];
  // Camera calibration instrinsics
  cam_intrinsics.resize(4);
  cv::FileNode n_instrin = fsSettings["intrinsics"];
  cam_intrinsics[0] = static_cast<double>(n_instrin["fx"]);
  cam_intrinsics[1] = static_cast<double>(n_instrin["fy"]);
  cam_intrinsics[2] = static_cast<double>(n_instrin["cx"]);
  cam_intrinsics[3] = static_cast<double>(n_instrin["cy"]);
  // Calculate boundary of feature measurement coordinate
  double fx = cam_intrinsics[0];
  double fy = cam_intrinsics[1];
  double cx = cam_intrinsics[2];
  double cy = cam_intrinsics[3];
  int U = cam_resolution[0];
  int V = cam_resolution[1];
  x_min = -cx/fx;
  y_min = -cy/fy;
  x_max = (U-cx)/fx;
  y_max = (V-cy)/fy;
  // Calculate grid height and width
  if (grid_rows*grid_cols!=0) {
      grid_width = (x_max-x_min)/grid_cols;
      grid_height = (y_max-y_min)/grid_rows;
  }
  else {
      grid_width = (x_max-x_min);
      grid_height = (y_max-y_min);
  }
  // Initialize the grid map
  for (int i=0; i<grid_rows*grid_cols; ++i)
      grid_map[i] = vector<FeatureIDType>(0);

  // Feature idp type
  feature_idp_dim = fsSettings["feature_idp_dim"];
  if (feature_idp_dim!=1 &&
    feature_idp_dim!=3) {
      cout << "Unknown type of feature idp type! Set as 3d idp." << endl;
      feature_idp_dim = 3;
  }

  // If apply Schmidt EKF
  use_schmidt = (static_cast<int>(fsSettings["use_schmidt"]) ? true : false);

  chi_square_threshold_feat = fsSettings["chi_square_threshold_feat"];

  dcampose_dimupose_fixed_flag = false; 

  // Print VIO setup
  cout << endl << "===========================================" << endl;
  if (if_FEJ_config)
      cout << "using FEJ..." << endl;
  else
      cout << "not using FEJ..." << endl;
  if (estimate_td)
      cout << "estimating td... initial td = " << state_server.td << endl;
  else
      cout << "not estimating td..." << endl;
  if (estimate_extrin)
      cout << "estimating extrinsic..." << endl;
  else
      cout << "not estimating extrinsic..." << endl;
  if (calib_imu)
      cout << "calibrating imu instrinsic online..." << endl;
  else
      cout << "not calibrating imu instrinsic online..." << endl;
  if (0==max_features*grid_rows*grid_cols)
      cout << "Pure MSCKF..." << endl;
  else {
      cout << "Hybrid MSCKF...Maximum number of feature in state is " << max_features*grid_rows*grid_cols << endl;
      if (1==feature_idp_dim)
          cout << "features augmented into state will use 1d idp" << endl;
      else
          cout << "features augmented into state will use 3d idp" << endl;
      if (use_schmidt)
          cout << "Applying Schmidt EKF" << endl;
  }

  if (if_ZUPT_valid == 1)
  {
    cout << "using zero velocity update" << endl;
  }
  else 
  {
    cout << "not using zero velocity update" << endl; 
  }

  if (use_left_perturbation_flag == 1)
  {
    cout << "using left perturbation" << endl;
  }
  else 
  {
    cout << "using right perturbation" << endl; 
  }

  if (use_larvio_flag || use_closed_form_cov_prop_flag)
  {
    cout << "using closed form covariance propagation" << endl;
  }
  else 
  {
    cout << "using Euler method covariance propagation" << endl;
  }

  if (use_larvio_flag)
  {
    cout << "using larvio" << endl;
  }
  else 
  {
    cout << "using orcvio" << endl;
  }
  if (initial_use_gt) {
      cout << "using ground-truth for initialization..." << endl;
      cout << "Initial Position: " << init_position(0) << "," << init_position(1) << "," << init_position(2) << endl;
      cout << "Initial Orientation: " << init_orientation(0) << "," << init_orientation(1)  
        << "," << init_orientation(2)  << "," << init_orientation(3) << endl;
      cout << "Initial Velocity: " << init_velocity(0) << "," << init_velocity(1) << "," << init_velocity(2) << endl;
      cout << "Initial ba: " << init_acc_bias(0) << "," << init_acc_bias(1) << "," << init_acc_bias(2) << endl;
      cout << "Initial bg: " << init_gyro_bias(0) << "," << init_gyro_bias(1) << "," << init_gyro_bias(2) << endl;
  }
  else
      cout << "using static/dynamic initialization..." << endl;

  if (prediction_only_flag) {
    cout << "using prediction only, no update" << endl;
  }

  cout << "===========================================" << endl << endl;

  return true;
}


bool OrcVIO::initialize() {
  if (!loadParameters()) return false;

  // debug log
  fImuState.open((output_dir+"state_est_geo_feat.txt").c_str(), ofstream::trunc);
  fTakeOffStamp.open((output_dir+"state_takeoff.txt").c_str(), ofstream::trunc);

  // Initialize state server
  if (use_larvio_flag || use_closed_form_cov_prop_flag)
  {

    state_server.continuous_noise_cov =
      Matrix<double, 12, 12>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) =
      Matrix3d::Identity()*imu_gyro_noise;
    state_server.continuous_noise_cov.block<3, 3>(3, 3) =
      Matrix3d::Identity()*imu_acc_noise;
    state_server.continuous_noise_cov.block<3, 3>(6, 6) =
      Matrix3d::Identity()*imu_gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(9, 9) =
      Matrix3d::Identity()*imu_acc_bias_noise;
  
  }
  else 
  {

    state_server.continuous_noise_cov =
      Matrix<double, 15, 15>::Zero();
    state_server.continuous_noise_cov.block<3, 3>(0, 0) =
      Matrix3d::Identity()*imu_gyro_noise;
    
    // note that the original orcvio has different order of variables in state 
    // compared with larvio 
    state_server.continuous_noise_cov.block<3, 3>(3, 3) =
      Matrix3d::Identity()*imu_acc_noise;
    // state_server.continuous_noise_cov.block<3, 3>(6, 6) =
    //   Matrix3d::Identity()*imu_acc_noise;
    
    state_server.continuous_noise_cov.block<3, 3>(9, 9) =
      Matrix3d::Identity()*imu_gyro_bias_noise;
    state_server.continuous_noise_cov.block<3, 3>(12, 12) =
      Matrix3d::Identity()*imu_acc_bias_noise;

  }

  // initialize if_FEJ flag
  if_FEJ = false;

  // initialize if_ZUPT flag
  if_ZUPT = false;

  // initialize bFirstFeatures flag
  bFirstFeatures = false;

  // initialize initializer
  flexInitPtr.reset(new FlexibleInitializer(
    zupt_max_feature_dis, Static_Num, state_server.td,
    state_server.Ma, state_server.Tg, state_server.As,
    sqrt(imu_acc_noise), sqrt(imu_acc_bias_noise),
    sqrt(imu_gyro_noise), sqrt(imu_gyro_bias_noise),
    state_server.imu_state.R_imu_cam0.transpose(),
    state_server.imu_state.t_cam0_imu, imu_img_timeTh));

    // Initialize the chi squared test table 
    chi_square_threshold_zupt = 0.95;
    chi_squre_num_threshold = 500;

    // https://github.com/KumarRobotics/msckf_vio/blob/050c50defa5a7fd9a04c1eed5687b405f02919b5/src/msckf_vio.cpp#L215-L221
    for (int i = 1; i < chi_squre_num_threshold; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table_feat[i] = boost::math::quantile(chi_squared_dist, chi_square_threshold_feat);
    }

    for (int i = 1; i < chi_squre_num_threshold; i++) {
        boost::math::chi_squared chi_squared_dist(i);
        chi_squared_table_zupt[i] = boost::math::quantile(chi_squared_dist, chi_square_threshold_zupt);
    }

  return true;
}


bool OrcVIO::processFeatures(MonoCameraMeasurementPtr msg,
        std::vector<ImuData>& imu_msg_buffer) {

  // features are not utilized until receiving imu msgs ahead
  if (!bFirstFeatures) {
      if ((imu_msg_buffer.begin() != imu_msg_buffer.end()) &&
          (imu_msg_buffer.begin()->timeStampToSec-msg->timeStampToSec-state_server.td <= 0.0))
          bFirstFeatures = true;
      else
          return false;
  }

  // Return if the gravity vector has not been set.
  if (!is_gravity_set) {
      if (initial_use_gt) {

        // initialize all IMU state
        state_server.imu_state.time = initial_state_time;
        state_server.imu_state.gyro_bias = init_gyro_bias;
        state_server.imu_state.acc_bias = init_acc_bias;
        state_server.imu_state.position = init_position;
        state_server.imu_state.velocity = init_velocity;
        state_server.imu_state.orientation = quaternionToRotation(init_orientation);

        // discard imu measurements earlier than the state time;
        int usefulImuSize = 0;
        for (const auto& imu_msg : imu_msg_buffer) {
          double imu_time = imu_msg.timeStampToSec;
          if (imu_time > initial_state_time) break;
          usefulImuSize++;
        }
        if (usefulImuSize>=imu_msg_buffer.size())
          usefulImuSize--;

        // Initialize last m_gyro and last m_acc
        const auto& imu_msg = imu_msg_buffer[usefulImuSize];
        m_gyro_old = imu_msg.angular_velocity;
        m_acc_old = imu_msg.linear_acceleration;

        // Earse used imu data
        imu_msg_buffer.erase(imu_msg_buffer.begin(),
        imu_msg_buffer.begin()+usefulImuSize);

        cout << "Success in initiating IMU using groundtruth." << endl;
      }
      else if (flexInitPtr->tryIncInit(imu_msg_buffer, msg,
              m_gyro_old, m_acc_old, state_server.imu_state)) {
        cout << "Success in initiating IMU using static/dynamic initialization." << endl;
      } else {
        return false;
      }

      is_gravity_set = true;
      // Set take off time
      take_off_stamp = state_server.imu_state.time;
      // Set last time of last ZUPT
      last_ZUPT_time = state_server.imu_state.time;
      // Initialize time of last update
      last_update_time = state_server.imu_state.time;
      // Update FEJ imu state
      state_server.imu_state_FEJ_now = state_server.imu_state;
      // debug log
      fTakeOffStamp << fixed << setprecision(9) << take_off_stamp << endl;	
  }

  // Propogate the IMU state.
  // that are received before the image msg.
  batchImuProcessing(msg->timeStampToSec+state_server.td, imu_msg_buffer);

  if (!prediction_only_flag)
  {
    // Add new observations for existing features or new
    // features in the map server.
    addFeatureObservations(msg);
  }

  // Augment the state vector. 
  stateAugmentation();

  // Check if a zero velocity update is happened
  if (if_ZUPT_valid)
  {

    if (if_use_feature_zupt_flag)
      if_ZUPT = checkZUPTFeat();
    else 
      if_ZUPT = checkZUPTIMU();
      
  }

  // Perform measurement update if necessary.
  removeLostFeatures();

  // Delete old imu state if necessary.
  pruneImuStateBuffer();

  // set if_FEJ flag if necessary
  if ( if_FEJ_config
      && !if_FEJ
      && state_server.imu_state.time-take_off_stamp>=0 ) {
      if_FEJ = true;
  }

  //==========================================================================
  //==========================================================================
  //==========================================================================
  // the pose logger in this function only considers the 
  // geometric feature update, but not the object residual update 

  // debug log: monitor imu state
  // double qw = state_server.imu_state.orientation(3);
  // double qx = state_server.imu_state.orientation(0);
  // double qy = state_server.imu_state.orientation(1);
  // double qz = state_server.imu_state.orientation(2);
  Eigen::Vector4d q = rotationToQuaternion(state_server.imu_state.orientation);
  double qw = q(3);
  double qx = q(0);
  double qy = q(1);
  double qz = q(2);

  double vx = state_server.imu_state.velocity(0);
  double vy = state_server.imu_state.velocity(1);
  double vz = state_server.imu_state.velocity(2);
  double px = state_server.imu_state.position(0);
  double py = state_server.imu_state.position(1);
  double pz = state_server.imu_state.position(2);
  double bgx = state_server.imu_state.gyro_bias(0);
  double bgy = state_server.imu_state.gyro_bias(1);
  double bgz = state_server.imu_state.gyro_bias(2);
  double bax = state_server.imu_state.acc_bias(0);
  double bay = state_server.imu_state.acc_bias(1);
  double baz = state_server.imu_state.acc_bias(2);
  Quaterniond qbc(state_server.imu_state.R_imu_cam0);
  double qbcw = qbc.w();
  double qbcx = qbc.x();
  double qbcy = qbc.y();
  double qbcz = qbc.z();
  double tx = state_server.imu_state.t_cam0_imu(0);
  double ty = state_server.imu_state.t_cam0_imu(1);
  double tz = state_server.imu_state.t_cam0_imu(2);
  
  // save the pose to txt for trajectory evaluation 
  // timestamp tx ty tz qx qy qz qw
  fImuState << state_server.imu_state.time-take_off_stamp << " "
      << px << " " << py << " " << pz << " "
      << qx << " " << qy << " " << qz << " " << qw << endl;

  // cout << "Timestamp: " << state_server.imu_state.time-take_off_stamp << endl;
  // cout << "Orientation: " <<  qw << "," << qx << "," << qy << "," << qz << endl;
  // cout << "Velocity: " << vx << "," << vy << "," << vz << endl;
  // cout << "Position: "  << px << " " << py << " " << pz << endl;
  // cout << "Gyro bias: " << bgx << " " << bgy << " " << bgz << endl;
  // cout << "Acc bias: " << bax << " " << bay << " " << baz << endl;
  // cout << "==============================================" << endl;

  // Update active_slam_features for visualization
  for (auto fid : state_server.feature_states) {
    active_slam_features[fid] = map_server[fid];
  }

  return true;
}


void OrcVIO::batchImuProcessing(const double& time_bound,
      std::vector<ImuData>& imu_msg_buffer) {
  // Counter how many IMU msgs in the buffer are used.
  int used_imu_msg_cntr = 0;

  meanSForce = Vector3d(0,0,0);
  int counter = 0;

  // time interval to the nearest image
  double dt = 0.0;

  // store the imu msg for zupt 
  imu_recent_zupt = {};

  for (const auto& imu_msg : imu_msg_buffer) {
    double imu_time = imu_msg.timeStampToSec;
    if (imu_time <= state_server.imu_state.time) {
      ++used_imu_msg_cntr;
      continue;
    }

    if ( imu_time-time_bound > imu_img_timeTh ) {
      break;   // threshold is adjusted according to the imu frequency
    }

    // save the imu msg for zupt 
    imu_recent_zupt.push_back(imu_msg);

    dt = imu_time-time_bound;

    // Convert the msgs.
    Vector3d m_gyro = imu_msg.angular_velocity;
    Vector3d m_acc = imu_msg.linear_acceleration;

    // Execute process model.
    processModel(imu_time, m_gyro, m_acc);  
    ++used_imu_msg_cntr;

    // update last m_gyro and last m_acc
    m_gyro_old = m_gyro;
    m_acc_old = m_acc;

    // add up specific forces
    meanSForce = meanSForce + m_acc;
    counter++;
  }

  // Set the state ID for the new IMU state.
  state_server.imu_state.id = IMUState::next_id++;   

  // Set time shift to the nearest image
  state_server.imu_state.dt = dt;

  // Remove all used IMU msgs.
  imu_msg_buffer.erase(imu_msg_buffer.begin(),
      imu_msg_buffer.begin()+used_imu_msg_cntr);

  meanSForce = meanSForce/counter;

  return;
}


void OrcVIO::processModel(const double& time,
    const Vector3d& m_gyro,
    const Vector3d& m_acc) {

  // Remove the bias from the measured gyro and acceleration
  IMUState& imu_state = state_server.imu_state;
  Vector3d f = m_acc-imu_state.acc_bias;  

  // unbiased acc  
  Vector3d acc = state_server.Ma*f;

  Vector3d w = m_gyro-state_server.As*acc-imu_state.gyro_bias;
  
  // unbiased gyro 
  Vector3d gyro = state_server.Tg*w;
  
  Vector3d f_old = m_acc_old-imu_state.acc_bias;
  Vector3d acc_old = state_server.Ma*f_old;
  Vector3d w_old = m_gyro_old-state_server.As*acc_old-imu_state.gyro_bias;
  Vector3d gyro_old = state_server.Tg*w_old;
  double dtime = time - imu_state.time;

  // Propagate the state 
  if (!use_larvio_flag)
  {
    // ref to https://github.com/moshanATucsd/orcvio_cpp/issues/1
    // we should always use acc in place (a_m - b_a) in Joan Sola's expressions and gyro in place of (w_m - b_m)
    predictNewStateOrcVIO(dtime, gyro, acc);
  }
  else 
  {
    predictNewStateLARVIO(dtime, gyro, acc);
  }

  // Compute error state transition matrix
  MatrixXd Phi;
  MatrixXd Q;

  // Propogate the state covariance matrix.
  if (use_larvio_flag || use_closed_form_cov_prop_flag)
  {
    calPhiClosedForm(Phi, dtime, f, w, acc, gyro, f_old, w_old, acc_old, gyro_old);
  }
  else
  {
    calPhiEulerMethod(Phi, dtime, acc, gyro);
  }

  // TODO: why use old state here?
  Matrix3d C_bk2w = state_server.imu_state_old.orientation;
  MatrixXd G = MatrixXd::Zero(LEG_DIM, 12);    

  if (use_larvio_flag || use_left_perturbation_flag)
  {
    G.block<3, 3>(0, 0) = -C_bk2w;
    G.block<3, 3>(3, 3) = -C_bk2w;
    G.block<3, 3>(9, 6) = Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Matrix3d::Identity();
  }
  else
  {
    // theta ~ omega
    G.block<3, 3>(0, 0) = -Matrix3d::Identity();
    // velocity ~ acc
    G.block<3, 3>(3, 3) = -C_bk2w;
    G.block<3, 3>(9, 6) = Matrix3d::Identity();
    G.block<3, 3>(12, 9) = Matrix3d::Identity();
  }

  // Closed form propagation
  Q = Phi * G * state_server.continuous_noise_cov *
      G.transpose() * Phi.transpose() * dtime;

  state_server.state_cov.block(0, 0, LEG_DIM, LEG_DIM) =  
    Phi*state_server.state_cov.block(0, 0, LEG_DIM, LEG_DIM)*Phi.transpose() + Q;

  if (state_server.state_cov.cols() > LEG_DIM) {	
    state_server.state_cov.block(
        0, LEG_DIM, LEG_DIM, state_server.state_cov.cols()-LEG_DIM) =
      Phi * state_server.state_cov.block(
        0, LEG_DIM, LEG_DIM, state_server.state_cov.cols()-LEG_DIM);
    state_server.state_cov.block(
        LEG_DIM, 0, state_server.state_cov.rows()-LEG_DIM, LEG_DIM) =
      state_server.state_cov.block(
        LEG_DIM, 0, state_server.state_cov.rows()-LEG_DIM, LEG_DIM) * Phi.transpose();
  }

  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;  	
  state_server.state_cov = state_cov_fixed;

  // Update the state info
  state_server.imu_state.time = time;      
  state_server.imu_state_FEJ_now.time = time;

  return;
}

void OrcVIO::predictNewStateLARVIO(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

  double gyro_norm = gyro.norm();
  Matrix4d Omega = Matrix4d::Zero();
  Omega.block<3, 3>(0, 0) = -skewSymmetric(gyro);
  Omega.block<3, 1>(0, 3) = gyro;
  Omega.block<1, 3>(3, 0) = -gyro;

  // Update old imu_state before it been propogated
  state_server.imu_state_old = state_server.imu_state;

  // Vector4d& q = state_server.imu_state.orientation;
  Vector4d q = rotationToQuaternion(state_server.imu_state.orientation);

  Vector3d& v = state_server.imu_state.velocity;
  Vector3d& p = state_server.imu_state.position;

  // Some pre-calculation
  Vector4d dq_dt, dq_dt2;
  if (gyro_norm > 1e-5) {
    dq_dt = (cos(gyro_norm*dt*0.5)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.5)*Omega) * q;	  
    dq_dt2 = (cos(gyro_norm*dt*0.25)*Matrix4d::Identity() +
      1/gyro_norm*sin(gyro_norm*dt*0.25)*Omega) * q;	
  }
  else {	
    dq_dt = (Matrix4d::Identity()+0.5*dt*Omega) *
      cos(gyro_norm*dt*0.5) * q;
    dq_dt2 = (Matrix4d::Identity()+0.25*dt*Omega) *
      cos(gyro_norm*dt*0.25) * q;
  }
  Matrix3d dR_dt_transpose = Quaterniond(dq_dt(3),dq_dt(0),dq_dt(1),dq_dt(2)).toRotationMatrix();
  Matrix3d dR_dt2_transpose = Quaterniond(dq_dt2(3),dq_dt2(0),dq_dt2(1),dq_dt2(2)).toRotationMatrix();

  // formula: k1 = f(tn, yn)
  Vector3d k1_v_dot = Quaterniond(q(3),q(0),q(1),q(2)).toRotationMatrix()*acc +
    IMUState::gravity;
  Vector3d k1_p_dot = v;
                    
  // formula: k2 = f(tn+dt/2, yn+k1*dt/2)
  Vector3d k1_v = v + k1_v_dot*dt/2;
  Vector3d k2_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k2_p_dot = k1_v;

  // formula: k3 = f(tn+dt/2, yn+k2*dt/2)
  Vector3d k2_v = v + k2_v_dot*dt/2;
  Vector3d k3_v_dot = dR_dt2_transpose*acc +
    IMUState::gravity;
  Vector3d k3_p_dot = k2_v;

  // formula: k4 = f(tn+dt, yn+k3*dt)
  Vector3d k3_v = v + k3_v_dot*dt;
  Vector3d k4_v_dot = dR_dt_transpose*acc +
    IMUState::gravity;
  Vector3d k4_p_dot = k3_v;

  // formula: yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
  q = dq_dt;				
  quaternionNormalize(q);	
  v = v + dt/6*(k1_v_dot+2*k2_v_dot+2*k3_v_dot+k4_v_dot);
  p = p + dt/6*(k1_p_dot+2*k2_p_dot+2*k3_p_dot+k4_p_dot);

  state_server.imu_state.orientation = quaternionToRotation(q);

  // Update FEJ imu state for last and present time
  state_server.imu_state_FEJ_old = state_server.imu_state_FEJ_now;
  state_server.imu_state_FEJ_now = state_server.imu_state;

  return;
}

void OrcVIO::predictNewStateOrcVIO(const double& dt,
    const Vector3d& gyro,
    const Vector3d& acc) {

    // Update old imu_state before it been propogated
    state_server.imu_state_old = state_server.imu_state;

    Matrix3d& R = state_server.imu_state.orientation;
    Vector3d& v = state_server.imu_state.velocity;
    Vector3d& p = state_server.imu_state.position;

    // update position 
    Matrix3d Hl = Hl_operator(dt*gyro);
    p = p + dt*v + IMUState::gravity*(pow(dt, 2)/2) + R * Hl * acc * pow(dt, 2);

    // update velocity
    Matrix3d Jl = Jl_operator(dt*gyro);
    v = v + IMUState::gravity*dt + R * Jl * acc * dt;

    // update rotation
    Sophus::SO3d R_temp = Sophus::SO3d::exp(dt*gyro);
    R = R*R_temp.matrix();

    // Update FEJ imu state for last and present time
    state_server.imu_state_FEJ_old = state_server.imu_state_FEJ_now;
    state_server.imu_state_FEJ_now = state_server.imu_state;

    return;

    }

void OrcVIO::stateAugmentation() {

  const Matrix3d& R_b2c =
      state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_b =
      state_server.imu_state.t_cam0_imu;

  // record the timestamp of current window 
  cur_window_timestamps.push_back(state_server.imu_state.time);

  // Add augmented IMU state
  state_server.imu_states_augment[state_server.imu_state.id] =
    IMUState_Aug(state_server.imu_state.id);
  IMUState_Aug& imu_state = state_server.imu_states_augment[
    state_server.imu_state.id];
  imu_state.time = state_server.imu_state.time;
  imu_state.dt = state_server.imu_state.dt;
  imu_state.orientation = state_server.imu_state.orientation;
  imu_state.position = state_server.imu_state.position;
  imu_state.position_FEJ = state_server.imu_state_FEJ_now.position;
  imu_state.R_imu_cam0 = R_b2c;
  imu_state.t_cam0_imu = t_c_b;

  // Add a new camera state to the state server.
  Matrix3d R_b2w = state_server.imu_state.orientation;

  Matrix3d R_w2b = R_b2w.transpose();
  Matrix3d R_w2c = R_b2c * R_w2b;
  Vector3d t_c_w =
      state_server.imu_state.position + R_b2w*t_c_b;
  imu_state.orientation_cam = R_w2c.transpose();
  imu_state.position_cam = t_c_w;

  // Update the covariance matrix of the state.
  MatrixXd J = MatrixXd::Zero(6, state_server.state_cov.rows());

  J.block(0, 0, 3, 3) = Matrix3d::Identity();
  J.block(3, 6, 3, 3) = Matrix3d::Identity();

  // Get old size
  size_t old_rows = state_server.state_cov.rows();
  size_t old_cols = state_server.state_cov.cols();

  // Compute new blocks.
  MatrixXd P12 = J * state_server.state_cov;
  MatrixXd P11 = P12 * J.transpose();

  // Resize the state covariance matrix.
  state_server.state_cov.conservativeResize(old_rows+6, old_cols+6);

  // Size of the covariance matrix without augmented features and nuisance states
  size_t feature_rows = feature_idp_dim*state_server.feature_states.size();
  size_t feature_cols = feature_rows;
  size_t nui_rows = 6*state_server.nui_ids.size();
  size_t nui_cols = nui_rows;
  size_t rest_rows = feature_rows + nui_rows;
  size_t rest_cols = rest_rows;
  size_t pose_rows = old_rows - rest_rows;
  size_t pose_cols = pose_rows;

  // Rename some matrix blocks
  const MatrixXd& P12_1 = P12.leftCols(pose_cols);
  const MatrixXd& P12_2 = P12.rightCols(rest_cols);

  // Move feature covariance
  state_server.state_cov.block(pose_rows+6, 0, rest_rows, old_cols) =
      state_server.state_cov.block(pose_rows, 0, rest_rows, old_cols).eval();
  state_server.state_cov.block(0, pose_cols+6, old_rows+6, rest_cols) =
      state_server.state_cov.block(0, pose_cols, old_rows+6, rest_cols).eval();

  // Fill in new blocks;
  state_server.state_cov.block(pose_rows, pose_cols, 6, 6) = P11;
  state_server.state_cov.block(pose_rows, 0, 6, pose_cols) = P12_1;
  state_server.state_cov.block(0, pose_cols, pose_rows, 6) = P12_1.transpose();
  state_server.state_cov.block(pose_rows, pose_cols+6, 6, rest_cols) = P12_2;
  state_server.state_cov.block(pose_rows+6, pose_cols, rest_rows, 6) = P12_2.transpose();

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov +
      state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}


void OrcVIO::addFeatureObservations(
    MonoCameraMeasurementPtr msg) {

  StateIDType state_id = state_server.imu_state.id;
  int curr_feature_num = map_server.size();
  int tracked_feature_num = 0;

  double dt = state_server.imu_state.dt;

  // Add new observations for existing features or new
  // features in the map server.
  for (const auto& feature : msg->features) {
    if (map_server.find(feature.id) == map_server.end()) {	
      // This is a new feature.
      map_server[feature.id] = Feature(feature.id);
      map_server[feature.id].observations[state_id] =    
        Vector2d(feature.u+feature.u_vel*dt, feature.v+feature.v_vel*dt);
      map_server[feature.id].observations_vel[state_id] =
        Vector2d(feature.u_vel, feature.v_vel);
      map_server[feature.id].totalObsNum++;
      if ( !(feature.u_init==-1 && feature.v_init==-1) &&
            state_server.imu_states_augment.find(state_id-1)!=state_server.imu_states_augment.end() ) {
        double dt_ = state_server.imu_states_augment[state_id-1].dt;
        map_server[feature.id].observations[state_id-1] =
          Vector2d(feature.u_init+feature.u_init_vel*dt_, feature.v_init+feature.v_init_vel*dt_);
        map_server[feature.id].observations_vel[state_id-1] =
          Vector2d(feature.u_init_vel, feature.v_init_vel);
        map_server[feature.id].totalObsNum++;
      }
    } else {
      // This is an old feature.
      map_server[feature.id].observations[state_id] =
        Vector2d(feature.u+feature.u_vel*dt, feature.v+feature.v_vel*dt);
      map_server[feature.id].observations_vel[state_id] =
        Vector2d(feature.u_vel, feature.v_vel);
      map_server[feature.id].totalObsNum++;
      ++tracked_feature_num;
      // push back feature distances of current and previous image
      if (if_ZUPT_valid && if_use_feature_zupt_flag && 
          map_server[feature.id].observations.find(state_id-1) != map_server[feature.id].observations.end()) {   
        Vector2d vec2d_c(feature.u, feature.v);
        Vector2d vec2d_p = map_server[feature.id].observations[state_id-1];
        coarse_feature_dis.push_back((vec2d_c-vec2d_p).norm());
      }
    }
  }

  tracking_rate =			
    static_cast<double>(tracked_feature_num) /
    static_cast<double>(curr_feature_num);

  return;
}


void OrcVIO::measurementJacobian_msckf(
        const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Matrix<double, 2, 6>& H_x, Matrix<double, 2, 6>& H_e, 
        Matrix<double, 2, 3>& H_f, Vector2d& r) {

  // Prepare all the required data.
  IMUState_Aug imu_state = state_server.imu_states_augment[state_id];
  const Feature& feature = map_server[feature_id];
  const Matrix3d& R_b2c = imu_state.R_imu_cam0;
  // t_cam0_imu is cam to imu 
  const Vector3d& t_c_b = imu_state.t_cam0_imu;

  // IMU pose for recover estimated feature position in camera frame
  Matrix3d R_b2w = imu_state.orientation;

  Matrix3d R_w2b = R_b2w.transpose();
  const Vector3d& t_b_w = imu_state.position;

  // Cam pose.
  Matrix3d R_w2c = R_b2c*R_w2b;
  Vector3d t_c_w = t_b_w + R_b2w*t_c_b;

  // 3d feature position in the world frame.
  // And its observation with the camera.
  const Vector3d& p_w = feature.position;   
  const Vector2d& z = feature.observations.find(state_id)->second;

  // Convert the feature position from the world frame to the cam frame.
  Vector3d p_cf_w = p_w - t_c_w;
  Vector3d p_c = R_w2c * p_cf_w;

  // Calculate the feature position wrt IMU in world frame
  // Vector3d p_bf_w = p_w-t_b_w;
  Vector3d p_bf_w = (if_FEJ ? p_w-imu_state.position_FEJ : p_w-t_b_w);    // FEJ in measurement Jacobian is applied here

  // Compute the Jacobians.
  Matrix<double, 2, 3> dz_dpc = Matrix<double, 2, 3>::Zero();
  dz_dpc(0, 0) = 1 / p_c(2);
  dz_dpc(1, 1) = 1 / p_c(2);
  dz_dpc(0, 2) = -p_c(0) / (p_c(2)*p_c(2));
  dz_dpc(1, 2) = -p_c(1) / (p_c(2)*p_c(2));

  Matrix<double, 3, 6> dpc_dxb = Matrix<double, 3, 6>::Zero();

  if (!use_larvio_flag)
  {
    // jacobian of inhomo coordinate wrt homo coordinate 
    Eigen::Matrix<double, 3, 4> temp_mat = Eigen::Matrix<double, 3, 4>::Zero();
    temp_mat.leftCols(3) = Eigen::Matrix<double, 3, 3>::Identity();

    Eigen::Matrix<double, 6, 6> dcampose_dimupose = Eigen::Matrix<double, 6, 6>::Zero();

    Eigen::Matrix<double, 4, 4> wTc = Eigen::Matrix<double, 4, 4>::Identity();
    wTc.block<3, 3>(0, 0) = R_w2c.transpose();
    wTc.block<3, 1>(0, 3) = t_c_w;

    Eigen::Vector4d uline_l = Eigen::Vector4d::Zero();
    uline_l(3) = 1;
    uline_l.head(3) = p_w;

    dcampose_dimupose = get_cam_wrt_imu_se3_jacobian(R_b2c, t_c_b, R_w2c, t_b_w, use_left_perturbation_flag);

    if (use_left_perturbation_flag)
    {
      dpc_dxb = temp_mat * wTc.inverse() * odotOperator(uline_l) * dcampose_dimupose;
    }
    else 
    {
      dpc_dxb = temp_mat * odotOperator(wTc.inverse() * uline_l) * dcampose_dimupose;
    }

    H_x = -dz_dpc*dpc_dxb;
  }
  else 
  {
    dpc_dxb.leftCols(3) = R_w2c * skewSymmetric(p_bf_w);
    dpc_dxb.rightCols(3) = -R_w2c;
    H_x = dz_dpc*dpc_dxb;
  }

  Matrix<double, 3, 6> dpc_dxe = Matrix<double, 3, 6>::Zero();
  dpc_dxe.leftCols(3) = (R_w2c * skewSymmetric(p_bf_w) * R_b2w)
      - (R_b2c * skewSymmetric(t_c_b));
  dpc_dxe.rightCols(3) = -R_b2c;

  Matrix3d dpc_dpw = R_w2c;

  // do not compute jacobian wrt extrinsic 
  H_e = dz_dpc*dpc_dxe;
  H_f = dz_dpc*dpc_dpw;

  // todo change to our residual definition and update EKF part 
  // Compute the residual.
  r = z - Vector2d(p_c(0)/p_c(2), p_c(1)/p_c(2));

  return;
}


void OrcVIO::featureJacobian_msckf(
    const FeatureIDType& feature_id,
    const std::vector<StateIDType>& state_ids,
    MatrixXd& H_x, VectorXd& r) {

  const auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_state_ids(0);
  for (const auto& state_id : state_ids) {   
    if (feature.observations.find(state_id) ==
        feature.observations.end()) continue;

    valid_state_ids.push_back(state_id);
  }

  int jacobian_row_size = 0;
  jacobian_row_size = 2 * valid_state_ids.size();  

  MatrixXd H_xj = MatrixXd::Zero(jacobian_row_size,
      state_server.state_cov.cols());
  MatrixXd H_fj = MatrixXd::Zero(jacobian_row_size, 3);
  VectorXd r_j = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  for (const auto& state_id : valid_state_ids) {   
    Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 6> H_ei = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
    Vector2d r_i = Vector2d::Zero();
    measurementJacobian_msckf(state_id, feature.id,
      H_xi, H_ei, H_fi, r_i);   

    auto state_iter = state_server.imu_states_augment.find(state_id);
    int state_cntr = std::distance(
            state_server.imu_states_augment.begin(), state_iter);  

    // Stack the Jacobians.
    H_xj.block<2, 6>(stack_cntr, LEG_DIM+6*state_cntr) = H_xi;
    H_xj.block<2, 6>(stack_cntr, 15) = H_ei;
    if (estimate_td)
      H_xj.block<2, 1>(stack_cntr, 21) = map_server[feature.id].observations_vel[state_id];
    H_fj.block<2, 3>(stack_cntr, 0) = H_fi;
    r_j.segment<2>(stack_cntr) = r_i;
    stack_cntr += 2;
  }

  // Project the residual and Jacobians onto the nullspace of H_fj.
  nullspace_project_inplace_svd(H_fj, H_xj, r_j);

  H_x = H_xj; 
  r = r_j;

  return;
}


void OrcVIO::measurementJacobian_ekf_3didp(const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 3>& H_f,
        Eigen::Matrix<double, 2, 6>& H_a,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 6>& H_e,
        Eigen::Vector2d& r) {

  // Prepare all the required data.
  if (state_server.imu_states_augment.find(state_id)
      ==state_server.imu_states_augment.end())
    printf("state_id is not in imu_states_augment !!!!!");
  IMUState_Aug imu_state_k = state_server.imu_states_augment[state_id];
  const Feature& feature = map_server[feature_id];

  bool anch_is_nui;
  IMUState_Aug imu_state_a;
  if (state_server.imu_states_augment.find(feature.id_anchor)
      !=state_server.imu_states_augment.end()) {
    imu_state_a = state_server.imu_states_augment[feature.id_anchor];
    anch_is_nui = false;
  } else {
    imu_state_a = state_server.nui_imu_states[feature.id_anchor];
    anch_is_nui = true;
    if (!use_schmidt)
      printf("shouldn't enter here if not using schmidt !!!!!");
  }
  const Matrix3d& R_b2c = imu_state_k.R_imu_cam0;
  const Vector3d& t_c_b = imu_state_k.t_cam0_imu;

  // IMU pose of camera coresponding to state_id.
  Matrix3d R_bk2w = imu_state_k.orientation;
  Matrix3d R_w2bk = R_bk2w.transpose();
  const Vector3d& t_bk_w = imu_state_k.position;
  // Camera pose of camera coresponding to state_id.
  Matrix3d R_w2ck = R_b2c*R_w2bk;
  Vector3d t_ck_w = t_bk_w + R_bk2w*t_c_b;

  // IMU pose of anchor camera.
  Matrix3d R_ba2w = imu_state_a.orientation;
  Matrix3d R_w2ba = R_ba2w.transpose();
  const Vector3d& t_ba_w = imu_state_a.position;
  // Camera pose of anchor camera
  Matrix3d R_w2ca;
  if (anch_is_nui) {
    Matrix3d R_ca2w = imu_state_a.orientation_cam;
    R_w2ca = R_ca2w.transpose();
  } else
    R_w2ca = R_b2c*R_w2ba;

  // Inverse depth param feature position in anchor camera frame
  const Vector3d& f_ca = feature.invParam;
  // 3D feature position in anchor camera frame
  Vector3d p_ca;
  if (if_FEJ && !anch_is_nui) {
    p_ca = R_b2c*(R_w2ba*(feature.position_FEJ-imu_state_a.position_FEJ)-t_c_b);
  } else {
    p_ca(0) = f_ca(0)/f_ca(2);
    p_ca(1) = f_ca(1)/f_ca(2);
    p_ca(2) = 1/f_ca(2);
  }

  // 3d feature position in the world frame.
  // And its observation with the camera.
  const Vector3d& p_w = feature.position;   
  const Vector2d& z = feature.observations.find(state_id)->second;

  // Convert the feature position from the world frame to the cam frame coresponding to state_id.
  Vector3d p_ckf_w = p_w - t_ck_w;
  Vector3d p_ck = R_w2ck * p_ckf_w;

  // Compute the residual
  r = z - Vector2d(p_ck(0)/p_ck(2), p_ck(1)/p_ck(2));

  // Compute the Jacobians
  if (state_id == feature.id_anchor) {
    H_f = Matrix<double, 2, 3>::Zero();
    H_f(0, 0) = 1;
    H_f(1, 1) = 1;
    H_a = Matrix<double, 2, 6>::Zero();
    H_x = Matrix<double, 2, 6>::Zero();
    H_e = Matrix<double, 2, 6>::Zero();
    return;
  }

  Matrix<double, 2, 3> J_k = Matrix<double, 2, 3>::Zero();
  J_k(0, 0) = 1 / p_ck(2);
  J_k(1, 1) = 1 / p_ck(2);
  J_k(0, 2) = -p_ck(0) / (p_ck(2)*p_ck(2));
  J_k(1, 2) = -p_ck(1) / (p_ck(2)*p_ck(2));

  Matrix3d J_p = R_w2ck * R_w2ca.transpose();

  Vector3d p_baf_w = ((if_FEJ&&!anch_is_nui) ?
    feature.position_FEJ-imu_state_a.position_FEJ : p_w-t_ba_w);
  Vector3d p_bkf_w = (if_FEJ ?
    feature.position_FEJ-imu_state_k.position_FEJ : p_w-t_bk_w);

  Matrix<double, 3, 6> J_xa = Matrix<double, 3, 6>::Zero();
  J_xa.leftCols(3) = -R_w2ck * skewSymmetric(p_baf_w);
  J_xa.rightCols(3) = R_w2ck;

  Matrix<double, 3, 6> J_xk = Matrix<double, 3, 6>::Zero();
  J_xk.leftCols(3) = R_w2ck * skewSymmetric(p_bkf_w);
  J_xk.rightCols(3) = -R_w2ck;

  Matrix<double, 3, 6> J_e = Matrix<double, 3, 6>::Zero();
  Matrix3d SkewMx = skewSymmetric(R_w2bk*p_bkf_w-t_c_b);
  Matrix3d Mx = R_w2bk * R_w2ba.transpose() * skewSymmetric(R_b2c.transpose()*p_ca);
  J_e.leftCols(3) = R_b2c * (SkewMx-Mx);
  J_e.rightCols(3) = R_b2c * (R_w2bk*R_w2ba.transpose() - Matrix3d::Identity());

  Matrix3d J_f = Matrix3d::Identity();
  J_f(0, 2) = -f_ca(0)/f_ca(2);
  J_f(1, 2) = -f_ca(1)/f_ca(2);
  J_f(2, 2) = -1/f_ca(2);
  J_f = J_f/f_ca(2);

  H_f = J_k*J_p*J_f;
  H_a = J_k*J_xa;
  H_x = J_k*J_xk;
  H_e = J_k*J_e;

  return;
}


void OrcVIO::measurementJacobian_ekf_1didp(const StateIDType& state_id,
        const FeatureIDType& feature_id,
        Eigen::Matrix<double, 2, 1>& H_f,
        Eigen::Matrix<double, 2, 6>& H_a,
        Eigen::Matrix<double, 2, 6>& H_x,
        Eigen::Matrix<double, 2, 6>& H_e,
        Eigen::Vector2d& r) {

  // Prepare all the required data.
  if (state_server.imu_states_augment.find(state_id)
      ==state_server.imu_states_augment.end())
    printf("state_id is not in imu_states_augment !!!!!");
  IMUState_Aug imu_state_k = state_server.imu_states_augment[state_id];
  const Feature& feature = map_server[feature_id];

  bool anch_is_nui;
  IMUState_Aug imu_state_a;
  if (state_server.imu_states_augment.find(feature.id_anchor)
    !=state_server.imu_states_augment.end()) {
    imu_state_a = state_server.imu_states_augment[feature.id_anchor];
    anch_is_nui = false;
  } else {
    imu_state_a = state_server.nui_imu_states[feature.id_anchor];
    anch_is_nui = true;
    if (!use_schmidt)
      printf("shouldn't enter here if not using schmidt !!!!!");
  }
  const Matrix3d& R_b2c = imu_state_k.R_imu_cam0;
  const Vector3d& t_c_b = imu_state_k.t_cam0_imu;

  // Corrected feature observation in anchor camera frame
  const Vector3d& f_an = feature.obs_anchor;

  // IMU pose of camera coresponding to state_id.
  Matrix3d R_bk2w = imu_state_k.orientation;
  Matrix3d R_w2bk = R_bk2w.transpose();
  const Vector3d& t_bk_w = imu_state_k.position;
  // Camera pose of camera coresponding to state_id.
  Matrix3d R_w2ck = R_b2c*R_w2bk;
  Vector3d t_ck_w = t_bk_w + R_bk2w*t_c_b;

  // IMU pose of anchor camera.
  Matrix3d R_ba2w = imu_state_a.orientation;

  Matrix3d R_w2ba = R_ba2w.transpose();
  const Vector3d& t_ba_w = imu_state_a.position;
  // Camera pose of anchor camera
  Matrix3d R_w2ca;
  if (anch_is_nui) {
    Matrix3d R_ca2w = imu_state_a.orientation_cam;
    R_w2ca = R_ca2w.transpose();
  } else
    R_w2ca = R_b2c*R_w2ba;

  // 3D feature position in anchor camera frame
  Vector3d p_ca;
  if (if_FEJ && !anch_is_nui) {
    p_ca = R_b2c*(R_w2ba*(feature.position_FEJ-imu_state_a.position_FEJ)-t_c_b);
  }else {
    p_ca(0) = f_an(0)/feature.invDepth;
    p_ca(1) = f_an(1)/feature.invDepth;
    p_ca(2) = 1/feature.invDepth;
  }

  // 3d feature position in the world frame.
  // And its observation with the camera.
  const Vector3d& p_w = feature.position;  
  const Vector2d& z = feature.observations.find(state_id)->second;

  // Convert the feature position from the world frame to the cam frame coresponding to state_id.
  Vector3d p_ckf_w = p_w - t_ck_w;
  Vector3d p_ck = R_w2ck * p_ckf_w;

  // Compute the residual
  r = z - Vector2d(p_ck(0)/p_ck(2), p_ck(1)/p_ck(2));

  // Compute the Jacobians
  if (state_id == feature.id_anchor) {
    printf("Measurement of anchor frame should not be used in 1d idp measurementJacobian!");
    r = Vector2d::Zero();
    H_f = Matrix<double, 2, 1>::Zero();
    H_a = Matrix<double, 2, 6>::Zero();
    H_x = Matrix<double, 2, 6>::Zero();
    H_e = Matrix<double, 2, 6>::Zero();
    return;
  }

  Matrix<double, 2, 3> J_k = Matrix<double, 2, 3>::Zero();
  J_k(0, 0) = 1 / p_ck(2);
  J_k(1, 1) = 1 / p_ck(2);
  J_k(0, 2) = -p_ck(0) / (p_ck(2)*p_ck(2));
  J_k(1, 2) = -p_ck(1) / (p_ck(2)*p_ck(2));

  Vector3d J_d = R_w2ck * R_w2ca.transpose() * f_an;

  Vector3d p_baf_w = ((if_FEJ&&!anch_is_nui) ?
    feature.position_FEJ-imu_state_a.position_FEJ : p_w-t_ba_w);
  Vector3d p_bkf_w = (if_FEJ ?
    feature.position_FEJ-imu_state_k.position_FEJ : p_w-t_bk_w);

  Matrix<double, 3, 6> J_xa = Matrix<double, 3, 6>::Zero();
  J_xa.leftCols(3) = -R_w2ck * skewSymmetric(p_baf_w);
  J_xa.rightCols(3) = R_w2ck;

  Matrix<double, 3, 6> J_xk = Matrix<double, 3, 6>::Zero();
  J_xk.leftCols(3) = R_w2ck * skewSymmetric(p_bkf_w);
  J_xk.rightCols(3) = -R_w2ck;

  Matrix<double, 3, 6> J_e = Matrix<double, 3, 6>::Zero();
  Matrix3d SkewMx = skewSymmetric(R_w2bk*p_bkf_w-t_c_b);
  Matrix3d Mx = R_w2bk * R_w2ba.transpose() * skewSymmetric(R_b2c.transpose()*p_ca);
  J_e.leftCols(3) = R_b2c * (SkewMx-Mx);
  J_e.rightCols(3) = R_b2c * (R_w2bk*R_w2ba.transpose() - Matrix3d::Identity());

  double J_rho = -1/(feature.invDepth*feature.invDepth);

  H_f = J_k*J_d*J_rho;
  H_a = J_k*J_xa;
  H_x = J_k*J_xk;
  H_e = J_k*J_e;

  return;
}


void OrcVIO::featureJacobian_ekf_new(const FeatureIDType& feature_id,
    const std::vector<StateIDType>& state_ids,
    Eigen::MatrixXd& H_x, Eigen::VectorXd& r) {
  
  auto& feature = map_server[feature_id];

  // Check how many camera states in the provided camera
  // id camera has actually seen this feature.
  vector<StateIDType> valid_state_ids(0);
  for (const auto& state_id : state_ids) {  
    if (feature.observations.find(state_id) ==
        feature.observations.end()) continue;

    if (1==feature_idp_dim &&
        state_id==feature.id_anchor) continue;    // Measurement of anchor frame should not be ussed when applying 1d idp

    valid_state_ids.push_back(state_id);
  }

  int jacobian_row_size = 2 * valid_state_ids.size();

  if (use_schmidt) {
    H_x = MatrixXd::Zero(jacobian_row_size,
        LEG_DIM+6*state_server.imu_states_augment.size()
        +feature_idp_dim*state_server.feature_states.size()
        +6*state_server.nui_ids.size());
  } else {
    H_x = MatrixXd::Zero(jacobian_row_size,
        LEG_DIM+6*state_server.imu_states_augment.size()
        +feature_idp_dim*state_server.feature_states.size());
  }
  r = VectorXd::Zero(jacobian_row_size);
  int stack_cntr = 0;

  auto anchor_iter = state_server.imu_states_augment.find(feature.id_anchor);
  int anchor_cntr = std::distance(
            state_server.imu_states_augment.begin(), anchor_iter);
  int anchor_index = LEG_DIM+6*anchor_cntr;

  auto feature_iter = find(state_server.feature_states.begin(),
            state_server.feature_states.end(), feature_id);
  int feature_cntr = std::distance(
            state_server.feature_states.begin(), feature_iter);
  int feature_index;
  if (use_schmidt) {
    int add = (LEG_DIM+6*state_server.imu_states_augment.size()
        +feature_idp_dim*state_server.feature_states.size()
        +6*state_server.nui_ids.size()) - state_server.state_cov.rows();
    int num_new = add/feature_idp_dim;
    int num_old = state_server.feature_states.size()-num_new;
    feature_index = state_server.state_cov.rows()+feature_idp_dim*(feature_cntr-num_old);
  } else
    feature_index = LEG_DIM+6*state_server.imu_states_augment.size()+feature_idp_dim*feature_cntr;

  for (const auto& state_id : valid_state_ids) {   
    auto state_iter = state_server.imu_states_augment.find(state_id);
    int imu_state_cntr = std::distance(
            state_server.imu_states_augment.begin(), state_iter);   

    Matrix<double, 2, 6> H_ai = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
    Matrix<double, 2, 6> H_ei = Matrix<double, 2, 6>::Zero();
    Vector2d r_i = Vector2d::Zero();

    if (3==feature_idp_dim) {
      Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
      measurementJacobian_ekf_3didp(state_id, feature_id,
        H_fi, H_ai, H_xi, H_ei, r_i);

      // Stack the Jacobians.
      H_x.block<2, 3>(stack_cntr, feature_index) = H_fi;
    } else {
      Matrix<double, 2, 1> H_fi = Matrix<double, 2, 1>::Zero();
      measurementJacobian_ekf_1didp(state_id, feature_id,
        H_fi, H_ai, H_xi, H_ei, r_i);

      // Stack the Jacobians.
      H_x.block<2, 1>(stack_cntr, feature_index) = H_fi;
    }

    // Stack the Jacobians.
    H_x.block<2, 6>(stack_cntr, anchor_index) = H_ai;
    H_x.block<2, 6>(stack_cntr, LEG_DIM+6*imu_state_cntr) = H_xi;
    H_x.block<2, 6>(stack_cntr, 15) = H_ei;
    if (estimate_td)
      H_x.block<2, 1>(stack_cntr, 21) = feature.observations_vel[state_id];
    r.segment<2>(stack_cntr) = r_i;
    stack_cntr += 2;
  }

  return;
}


void OrcVIO::featureJacobian_ekf(const FeatureIDType& feature_id,
    Eigen::MatrixXd& H_x, Eigen::Vector2d& r) {
  
  auto& feature = map_server[feature_id];

  const StateIDType& imu_id = state_server.imu_state.id;

  if (1==feature_idp_dim && imu_id==feature.id_anchor)
      printf("Measurement of anchor frame should not appear in featureJacobian_ekf");

  int anchor_cntr;
  int anchor_index;
  int num_new = (LEG_DIM+6*state_server.imu_states_augment.size()
        +feature_idp_dim*state_server.feature_states.size()
        +6*state_server.nui_ids.size()
        -state_server.state_cov.rows())/feature_idp_dim;
  if (use_schmidt &&
      find(state_server.nui_ids.begin(), state_server.nui_ids.end(), feature.id_anchor)
      != state_server.nui_ids.end()) {
    auto anchor_iter = find(
        state_server.nui_ids.begin(), state_server.nui_ids.end(), feature.id_anchor);
    anchor_cntr = std::distance(
        state_server.nui_ids.begin(), anchor_iter);
    anchor_index = LEG_DIM
        +6*state_server.imu_states_augment.size()
        +feature_idp_dim*(state_server.feature_states.size()-num_new)
        +6*anchor_cntr;
  } else if (state_server.imu_states_augment.find(feature.id_anchor)!=state_server.imu_states_augment.end()) {
    auto anchor_iter = state_server.imu_states_augment.find(feature.id_anchor);
    anchor_cntr = std::distance(
        state_server.imu_states_augment.begin(), anchor_iter);
    anchor_index = LEG_DIM+6*anchor_cntr;
  } else {
    printf("ERROR HAPPENED IN J_EKF !!!!");
  }

  auto feature_iter = find(state_server.feature_states.begin(),
            state_server.feature_states.end(), feature_id);
  int feature_cntr = std::distance(
            state_server.feature_states.begin(), feature_iter);
  int feature_index = LEG_DIM+6*state_server.imu_states_augment.size()+feature_idp_dim*feature_cntr;

  auto state_iter = state_server.imu_states_augment.find(imu_id);
  int state_cntr = std::distance(
          state_server.imu_states_augment.begin(), state_iter);
  int state_index = LEG_DIM+6*state_cntr;

  H_x = MatrixXd::Zero(2, state_server.state_cov.cols());

  Matrix<double, 2, 6> H_ai = Matrix<double, 2, 6>::Zero();
  Matrix<double, 2, 6> H_xi = Matrix<double, 2, 6>::Zero();
  Matrix<double, 2, 6> H_ei = Matrix<double, 2, 6>::Zero();
  Vector2d r_i = Vector2d::Zero();

  if (3==feature_idp_dim) {
    Matrix<double, 2, 3> H_fi = Matrix<double, 2, 3>::Zero();
    measurementJacobian_ekf_3didp(imu_id, feature_id,
      H_fi, H_ai, H_xi, H_ei, r_i);

    H_x.block<2, 3>(0, feature_index) = H_fi;
  } else {
    Matrix<double, 2, 1> H_fi = Matrix<double, 2, 1>::Zero();
    measurementJacobian_ekf_1didp(imu_id, feature_id,
      H_fi, H_ai, H_xi, H_ei, r_i);

    H_x.block<2, 1>(0, feature_index) = H_fi;
  }

  H_x.block<2, 6>(0, anchor_index) = H_ai;
  H_x.block<2, 6>(0, state_index) = H_xi;
  H_x.block<2, 6>(0, 15) = H_ei;
  if (estimate_td)
    H_x.block<2, 1>(0, 21) = feature.observations_vel[imu_id];
  r = r_i;

  return;
}


void OrcVIO::measurementUpdate_msckf(
    const MatrixXd& H, const VectorXd& r) {

  if (H.rows() == 0 || r.rows() == 0) return;

  // Decompose the final Jacobian matrix to reduce computational
  // complexity as in Equation (28), (29).
  MatrixXd H_thin;
  VectorXd r_thin;

  if (H.rows() > H.cols()) {  
    // Convert H to a sparse matrix.
    SparseMatrix<double> H_sparse = H.sparseView();

    // Perform QR decompostion on H_sparse.
    SPQR<SparseMatrix<double> > spqr_helper;
    spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
    spqr_helper.compute(H_sparse);

    MatrixXd H_temp;
    VectorXd r_temp;
    (spqr_helper.matrixQ().transpose() * H).evalTo(H_temp);
    (spqr_helper.matrixQ().transpose() * r).evalTo(r_temp);

    H_thin = H_temp.topRows(LEG_DIM+state_server.imu_states_augment.size()*6);   
    r_thin = r_temp.head(LEG_DIM+state_server.imu_states_augment.size()*6);
  } else {
    H_thin = H;
    r_thin = r;
  }

  // Compute the Kalman gain.
  MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_thin*P*H_thin.transpose() +
      feature_observation_noise*MatrixXd::Identity(
        H_thin.rows(), H_thin.rows());
  MatrixXd K_transpose = S.ldlt().solve(H_thin*P); 
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r_thin;

  // increment IMU and camera state 
  incrementState_IMUCam(delta_x);

  // Update the augmented feature states
  int base_cntr = LEG_DIM+6*state_server.imu_states_augment.size();
  auto feature_itr = state_server.feature_states.begin();
  for (int i = 0; i < state_server.feature_states.size();
    ++i, feature_itr++) {
    FeatureIDType feature_id = (*feature_itr);
    IMUState_Aug imu_state_aug;
    if (use_schmidt && find(state_server.nui_ids.begin(), state_server.nui_ids.end(),
        map_server[feature_id].id_anchor)!=state_server.nui_ids.end())
      imu_state_aug = state_server.nui_imu_states[map_server[feature_id].id_anchor];
    else if (state_server.imu_states_augment.find(map_server[feature_id].id_anchor)
        != state_server.imu_states_augment.end())
      imu_state_aug = state_server.imu_states_augment[map_server[feature_id].id_anchor];
    else
      printf("ERROR HAPPENED IN updt_msckf");
    Matrix3d R_c2w = imu_state_aug.orientation_cam;
    const Vector3d& t_c_w = imu_state_aug.position_cam;
    Vector3d p_c;
    if (3==feature_idp_dim) {
      // Update invParam
      const Vector3d& delta_f_aug = delta_x.segment<3>(base_cntr+i*3);
      map_server[feature_id].invParam += delta_f_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].invParam(0)/map_server[feature_id].invParam(2);
      p_c(1) = map_server[feature_id].invParam(1)/map_server[feature_id].invParam(2);
      p_c(2) = 1/map_server[feature_id].invParam(2);
    } else {
      // Update invDepth
      double delta_rho_aug = delta_x(base_cntr+i);
      map_server[feature_id].invDepth += delta_rho_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].obs_anchor(0)/map_server[feature_id].invDepth;
      p_c(1) = map_server[feature_id].obs_anchor(1)/map_server[feature_id].invDepth;
      p_c(2) = 1/map_server[feature_id].invDepth;
    }
    Vector3d p_w = R_c2w*p_c + t_c_w;
    map_server[feature_id].position = p_w;
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_thin.cols()) - K*H_thin;  
  if (use_schmidt && state_server.nui_ids.size()>0) {
    MatrixXd P_nui = P.block(
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size());
    P = I_KH*P;
    P.block(
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size()) = P_nui;
  } else {
    P = I_KH*P;  
  }

  // Fix the covariance to be symmetric
  P = ((P + P.transpose()) / 2.0).eval();

  // reset First Estimate Points if last update is far away
  // if ( state_server.imu_state.time-last_update_time>reset_fej_threshold )
  //   resetFejPoint();
  last_update_time = state_server.imu_state.time;

  return;
}


void OrcVIO::measurementUpdate_hybrid(
        const Eigen::MatrixXd& H_ekf_new, const Eigen::VectorXd& r_ekf_new, 
        const Eigen::MatrixXd& H_ekf, const Eigen::VectorXd& r_ekf,
        const Eigen::MatrixXd& H_msckf, const Eigen::VectorXd& r_msckf) {
  
  // Size of new EKF-SLAM features.
  int sz_new = H_ekf_new.cols() - state_server.state_cov.cols();
  // Size of residuals
  int sz_r = r_ekf_new.rows() + r_ekf.rows() + r_msckf.rows();
  if (0==sz_r)  return;

  // Stacked Jacobians
  // Ho
  MatrixXd H_o = MatrixXd::Zero(sz_r-sz_new, state_server.state_cov.cols());
  H_o.block(
    0, 0,
    H_msckf.rows(), state_server.state_cov.cols()) = H_msckf;
  H_o.block(
    H_msckf.rows(), 0,
    H_ekf.rows(), state_server.state_cov.cols()) = H_ekf;
  H_o.block(
    H_msckf.rows()+H_ekf.rows(), 0,
    H_ekf_new.rows()-sz_new, state_server.state_cov.cols())
      = H_ekf_new.block(0, 0, H_ekf_new.rows()-sz_new, state_server.state_cov.cols());
  // H1
  MatrixXd H_1
    = H_ekf_new.block(
        H_ekf_new.rows()-sz_new, 0, sz_new, state_server.state_cov.cols());
  // H2
  MatrixXd H_2
    = H_ekf_new.block(
        H_ekf_new.rows()-sz_new, state_server.state_cov.cols(), sz_new, sz_new);

  // Stacked residuals
  // ro
  VectorXd r_o = VectorXd::Zero(sz_r-sz_new);
  r_o.segment(0, r_msckf.rows()) = r_msckf;
  r_o.segment(r_msckf.rows(), r_ekf.rows()) = r_ekf;
  r_o.segment(
    r_msckf.rows()+r_ekf.rows(), r_ekf_new.rows()-sz_new)
      = r_ekf_new.segment(0, r_ekf_new.rows()-sz_new);
  // r1
  VectorXd r_1
    = r_ekf_new.segment(r_ekf_new.rows()-sz_new, sz_new);

  // Compute the Kalman gain of legacy state.
  MatrixXd& P = state_server.state_cov;
  MatrixXd S = H_o*P*H_o.transpose() +
      feature_observation_noise*MatrixXd::Identity(
        H_o.rows(), H_o.rows());
  MatrixXd K_transpose = S.ldlt().solve(H_o*P);
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the legacy state.
  VectorXd dx_leg = K * r_o;

  // Compute the error of new added state.
  VectorXd dx_new;
  MatrixXd HH;
  if (r_1.rows()>0) {
    HH = H_2.ldlt().solve(H_1);
    dx_new = -HH*dx_leg + H_2.ldlt().solve(r_1);
  }

  // Compute the error of the state.
  VectorXd delta_x;
  if (r_1.rows()>0) {
    delta_x = VectorXd::Zero(dx_leg.rows()+dx_new.rows());
    delta_x.bottomRows(dx_new.rows()) = dx_new;
  } else
    delta_x = VectorXd::Zero(dx_leg.rows());
  delta_x.topRows(dx_leg.rows()) = dx_leg;

  // increment IMU and camera state 
  incrementState_IMUCam(delta_x);

  // Update the augmented feature states
  int base_cntr = LEG_DIM+6*state_server.imu_states_augment.size();
  auto feature_itr = state_server.feature_states.begin();
  int num_new = sz_new/feature_idp_dim;
  int num_old = state_server.feature_states.size()-num_new;
  for (int i = 0; i < state_server.feature_states.size();
    ++i, feature_itr++) {
    FeatureIDType feature_id = (*feature_itr);
    IMUState_Aug imu_state_aug;
    if (use_schmidt && find(state_server.nui_ids.begin(), state_server.nui_ids.end(),
        map_server[feature_id].id_anchor)!=state_server.nui_ids.end())
      imu_state_aug = state_server.nui_imu_states[map_server[feature_id].id_anchor];
    else if (state_server.imu_states_augment.find(map_server[feature_id].id_anchor)
        != state_server.imu_states_augment.end())
      imu_state_aug = state_server.imu_states_augment[map_server[feature_id].id_anchor];
    else
      printf("ERROR HAPPENED IN updt_hybrid");
    Matrix3d R_c2w = imu_state_aug.orientation_cam;
    const Vector3d& t_c_w = imu_state_aug.position_cam;
    Vector3d p_c;
    if (3==feature_idp_dim) {
      // Update invParam
      Vector3d delta_f_aug;
      if (use_schmidt && i>=num_old)
        delta_f_aug = delta_x.segment<3>(state_server.state_cov.rows()+(i-num_old)*3);
      else
        delta_f_aug = delta_x.segment<3>(base_cntr+i*3);
      map_server[feature_id].invParam += delta_f_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].invParam(0)/map_server[feature_id].invParam(2);
      p_c(1) = map_server[feature_id].invParam(1)/map_server[feature_id].invParam(2);
      p_c(2) = 1/map_server[feature_id].invParam(2);
    } else {
      // Update invDepth
      double delta_rho_aug;
      if (use_schmidt && i>=num_old) {
        delta_rho_aug = delta_x(state_server.state_cov.rows()+i-num_old);
      } else
        delta_rho_aug = delta_x(base_cntr+i);
      map_server[feature_id].invDepth += delta_rho_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].obs_anchor(0)/map_server[feature_id].invDepth;
      p_c(1) = map_server[feature_id].obs_anchor(1)/map_server[feature_id].invDepth;
      p_c(2) = 1/map_server[feature_id].invDepth;
    }
    Vector3d p_w = R_c2w*p_c + t_c_w;
    map_server[feature_id].position = p_w;
  }

  // Update legacy state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H_o.cols()) - K*H_o;
  if (use_schmidt && state_server.nui_ids.size()>0) {
    MatrixXd P_nui = P.block(
      base_cntr+feature_idp_dim*num_old,
      base_cntr+feature_idp_dim*num_old,
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size());
    P = I_KH*P;
    P.block(base_cntr+feature_idp_dim*num_old,
      base_cntr+feature_idp_dim*num_old,
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size()) = P_nui;
  } else {
    P = I_KH*P;
  }

  // Fix the covariance to be symmetric
  P = ((P + P.transpose()) / 2.0).eval();

  // Update augmented state covariance.
  if (r_1.rows()>0) {
    MatrixXd nHHP = -HH * P;
    MatrixXd H22 = H_2.transpose() * H_2;
    MatrixXd P22 = -nHHP*HH.transpose()
      + feature_observation_noise*(H22.ldlt().solve(MatrixXd::Identity(sz_new,sz_new)));

    // Resize the state covariance matrix.
    size_t old_rows = P.rows();
    size_t old_cols = P.cols();
    P.conservativeResize(old_rows+sz_new, old_cols+sz_new);
    if (use_schmidt && state_server.nui_ids.size()>0) {
      size_t nui_rows = 6*state_server.nui_ids.size();
      size_t nui_cols = nui_rows;
      // Move nuisance covariance blocks
      P.block(old_rows+sz_new-nui_rows, 0, nui_rows, old_cols) =
          P.block(old_rows-nui_rows, 0, nui_rows, old_cols).eval();
      P.block(0, old_cols+sz_new-nui_cols, old_rows+sz_new, nui_cols) =
          P.block(0, old_cols-nui_cols, old_rows+sz_new, nui_cols).eval();
      // Fill in new blocks
      P.block(old_rows-nui_rows, 0, sz_new, old_cols-nui_cols) = nHHP.leftCols(old_cols-nui_cols);
      P.block(old_rows-nui_rows, old_cols+sz_new-nui_cols, sz_new, nui_cols) = nHHP.rightCols(nui_cols);
      P.block(0, old_cols-nui_cols, old_rows-nui_rows, sz_new) = nHHP.leftCols(old_cols-nui_cols).transpose();
      P.block(old_rows+sz_new-nui_rows, old_cols-nui_cols, nui_rows, sz_new) = nHHP.rightCols(nui_cols).transpose();
      P.block(old_rows-nui_rows, old_cols-nui_cols, sz_new, sz_new) = P22;
    } else {
      P.block(old_rows, 0, sz_new, old_cols) = nHHP;
      P.block(0, old_cols, old_rows, sz_new) = nHHP.transpose();
      P.block(old_rows, old_cols, sz_new, sz_new) = P22;
    }

    // Fix the covariance to be symmetric
    P = ((P + P.transpose()) / 2.0).eval();
  }

  // reset First Estimate Points if last update is far away
  // if ( state_server.imu_state.time-last_update_time>reset_fej_threshold )
  //   resetFejPoint();
  last_update_time = state_server.imu_state.time;

  return;
}


bool OrcVIO::gatingTestFeature(
    const MatrixXd& H, const VectorXd& r, const int& dof) {

  MatrixXd P1 = H * state_server.state_cov * H.transpose();
  MatrixXd P2 = feature_observation_noise *
    MatrixXd::Identity(H.rows(), H.rows());
  double gamma = r.transpose() * (P1+P2).ldlt().solve(r);  

  double chi2_check;
  if(dof < chi_squre_num_threshold) {
      chi2_check = chi_squared_table_feat[dof];
  } else {
      boost::math::chi_squared chi_squared_dist(dof);
      chi2_check = boost::math::quantile(chi_squared_dist, chi_square_threshold_feat);
      printf("[Gating test]: chi2_check over the residual limit - %d\n", (int)dof);
  }

  if (gamma < chi2_check) {
    // cout << "passed" << endl;
    return true;
  } else {
    // cout << "failed" << endl;
    return false;
  }

  // the implementation below follows openvins 
  // ref https://github.com/rpng/open_vins/blob/e169a77906941d662597bc83c8552e28b9a404f4/ov_msckf/src/update/UpdaterMSCKF.cpp

  // /// Chi2 distance check
  // double sigma_pix_sq = 1.0; 

  // Eigen::MatrixXd P_marg = state_server.state_cov;
  // Eigen::MatrixXd S = H * P_marg * H.transpose();
  // S.diagonal() += sigma_pix_sq * Eigen::VectorXd::Ones(S.rows());
  // double chi2 = r.dot(S.llt().solve(r));

  // // Get our threshold (we precompute up to chi_squre_num_threshold but handle the case that it is more)
  // double chi2_check;
  // if(r.rows() < chi_squre_num_threshold) {
  //     chi2_check = chi_squared_table[r.rows()];
  // } else {
  //     boost::math::chi_squared chi_squared_dist(r.rows());
  //     chi2_check = boost::math::quantile(chi_squared_dist, 0.95);
  //     printf("[Gating test]: chi2_check over the residual limit - %d\n", (int)r.rows());
  // }

  // // Check if we should delete or not
  // int chi2_multipler = 1;
  
  // if(chi2 > chi2_multipler*chi2_check) {
  //   printf("[Gating test]: fail\n");
  //   return false;
  // }
  // else 
  // {
  //   printf("[Gating test]: pass\n");
  //   return true;
  // }

}

// use jacobian_wrt_sensor_state to construct 
// jacobian wrt states in the current window 
// note Hx is a place holder 
bool OrcVIO::constructObjectResidualJacobians(const Eigen::MatrixXd& jacobian_wrt_sensor_state, const std::vector<double>& object_timestamps,
      Eigen::MatrixXd& Hx, Eigen::MatrixXd& Hf, Eigen::VectorXd& res, const std::vector<int>& zs_num_wrt_timestamps, const MatrixXd& valid_camera_pose_mat) 
{

  // need to check whether there is at least one pose in current window 
  // if no pose in window, then return false and do not update 
  bool has_pose_in_window_flag = false; 

  if (!use_object_residual_update_cam_pose_flag)
  {
    // std::cout << "not using object feature for camera pose update" << std::endl; 
    return has_pose_in_window_flag;
  }

  // check the input
  const int total_obs_num = object_timestamps.size(); 
  // dof of camera pose should be 6 
  assert(jacobian_wrt_sensor_state.cols() == 6);
  // total number of observations should be same for state and object 
  assert(jacobian_wrt_sensor_state.rows() == Hf.rows());
  // dof of camera pose should be 6
  assert(valid_camera_pose_mat.rows() == 6);
  // total number of camera poses should be same with number of timestamps 
  assert(valid_camera_pose_mat.cols() == total_obs_num);

  const int sensor_state_dim = state_server.state_cov.cols();
  const int object_state_dim = Hf.cols();

  // get the number of total valid keypoint observations 
  int sum_of_zs_num = 0;
  for (auto& n : zs_num_wrt_timestamps)
      sum_of_zs_num += 2*n;

  Hx = Eigen::MatrixXd::Zero(jacobian_wrt_sensor_state.rows(), sensor_state_dim); 
  Eigen::MatrixXd Hf_temp = Hf;
  Eigen::MatrixXd res_temp = res; 

  int Hx_zs_last_row_index = 0; 
  int fjac_zs_last_row_index = 0; 
  int pose_index, zs_num;
  double timestamp;

  // for debugging 
  // for (auto const& tm: cur_window_timestamps)
  //   std::cout << "cur_window_timestamps " << tm << std::endl;

  for (int timestamp_count = 0; timestamp_count < total_obs_num; timestamp_count++)
  {

    timestamp = object_timestamps[timestamp_count];

    // std::cout << "object timestamp " << timestamp << std::endl;

    zs_num = zs_num_wrt_timestamps[timestamp_count];
    const int zs_num_frame = zs_num * 2;

    std::vector<double>::iterator it = std::find(cur_window_timestamps.begin(), cur_window_timestamps.end(), timestamp);
    if (it != cur_window_timestamps.end())
    {
      // std::cout << "Element Found" << std::endl;

      // get jacobian of camera pose wrt imu pose 
      if (!dcampose_dimupose_fixed_flag)
      {
        // get camera pose 
        Eigen::Matrix<double, 6, 1> se3 = valid_camera_pose_mat.col(timestamp_count);
        Eigen::Matrix4d wTc = Sophus::SE3d::exp(se3).matrix();
        // get the imu pose 
        const Matrix3d& R_b2c =
            state_server.imu_state.R_imu_cam0;
        // t_cam0_imu is cam to imu 
        const Vector3d& t_c_b =
            state_server.imu_state.t_cam0_imu;
        Eigen::Matrix<double, 3, 3> R_w2c = wTc.inverse().block<3,3>(0, 0);
        Eigen::Vector3d t_b_w = wTc.block<3,3>(0,0) * (-R_b2c * t_c_b) + wTc.block<3,1>(0,3);
        dcampose_dimupose_mat = get_cam_wrt_imu_se3_jacobian(R_b2c, t_c_b, R_w2c, t_b_w, use_left_perturbation_flag);
      }

      // Get index of element from iterator
      pose_index = std::distance(cur_window_timestamps.begin(), it);

      // for jacobian wrt IMU state 
      Hx.block(Hx_zs_last_row_index, LEG_DIM+6*pose_index, zs_num_frame, 6) = 
        jacobian_wrt_sensor_state.block(fjac_zs_last_row_index, 0, zs_num_frame, 6)
        * dcampose_dimupose_mat;

      // for jacobian wrt object state
      Hf_temp.block(Hx_zs_last_row_index, 0, zs_num_frame, object_state_dim) = 
        Hf.block(fjac_zs_last_row_index, 0, zs_num_frame, object_state_dim);

      // for semantic keypoints residual
      res_temp.block(Hx_zs_last_row_index, 0, zs_num_frame, 1) = 
        res.block(fjac_zs_last_row_index, 0, zs_num_frame, 1);

      Hx_zs_last_row_index += zs_num_frame;

      // for jacobian wrt IMU state
      Hx.block(Hx_zs_last_row_index, LEG_DIM+6*pose_index, 4, 6) = 
        jacobian_wrt_sensor_state.block(sum_of_zs_num+timestamp_count*4, 0, 4, 6)
        * dcampose_dimupose_mat;

      // for jacobian wrt object state
      Hf_temp.block(Hx_zs_last_row_index, 0, 4, object_state_dim) = 
        Hf.block(sum_of_zs_num+timestamp_count*4, 0, 4, object_state_dim);

      // for bounding box residual
      res_temp.block(Hx_zs_last_row_index, 0, 4, 1) = 
        res.block(sum_of_zs_num+timestamp_count*4, 0, 4, 1);

      Hx_zs_last_row_index += 4;

      fjac_zs_last_row_index += zs_num_frame;

      has_pose_in_window_flag = true; 

    }
    else
    {
      // std::cout << "Element Not Found" << std::endl;

      fjac_zs_last_row_index += zs_num_frame;
    }

  }

  Hf = Hf_temp;
  res = res_temp;

  Hx.conservativeResize(Hx_zs_last_row_index, sensor_state_dim);
  Hf.conservativeResize(Hx_zs_last_row_index, object_state_dim);
  res.conservativeResize(Hx_zs_last_row_index, 1);

  return has_pose_in_window_flag; 

}


void OrcVIO::removeLostObjects(Eigen::MatrixXd &H_x, Eigen::MatrixXd &H_f, Eigen::VectorXd &res) 
{

  if (H_x.rows() == 0 || res.rows() == 0) return;

  if (!use_object_residual_update_cam_pose_flag) return; 

  // do nullspace trick 
  bool nullspace_trick_success_flag = nullspace_project_inplace_svd(H_f, H_x, res);

  if (!nullspace_trick_success_flag)
  {
    std::cout << "[OrcVIO] Object feature nullspace trick fails" << std::endl;
    return; 
  }

  // do geting test 
  // dof is just res.rows() after nullspace trick 
  if (!gatingTestFeature(H_x, res, res.rows()))
  {
    std::cout << "[OrcVIO] Object feature gating test fails" << std::endl;
    return; 
  }

  if (!check_nan(H_x) || !check_nan(res))
  {
    std::cout << "[OrcVIO] Object feature jacobian or residual contains nan" << std::endl;
    return; 
  }

  std::cout << "[OrcVIO] Use object feature for sensor state update" << std::endl;

  // std::cout << "H_x " << H_x << std::endl; 
  // std::cout << "res " << res << std::endl; 

  // do update use measurementUpdate_msckf
  // note that we just use feature observation noise for all residuals 
  measurementUpdate_msckf(H_x, res); 

}


void OrcVIO::removeLostFeatures() {
  // Remove the features that lost track.
  // BTW, find the size the final Jacobian matrix and residual vector.
  int jacobian_row_size_msckf = 0;
  vector<FeatureIDType> invalid_feature_ids(0);
  vector<FeatureIDType> msckf_feature_ids(0);
  vector<FeatureIDType> msckf_lost_feature_ids(0);
  int jacobian_row_size_ekf_new = 0;
  vector<FeatureIDType> ekf_new_feature_ids(0);
  vector<FeatureIDType> ekf_lost_feature_ids(0);
  int jacobian_row_size_ekf = 0;
  vector<FeatureIDType> ekf_feature_ids(0);

  // Loop for features in state
  for (auto iter = map_server.begin();  
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // TRUE if this feature is tracked now
    bool if_tracked_now =
          feature.observations.find(state_server.imu_state.id)!=feature.observations.end();

    // this block is only used for hybrid msckf 
    if (feature.in_state) {
      if (if_tracked_now) {
        jacobian_row_size_ekf += 2;
        ekf_feature_ids.push_back(feature.id);
      } else {
        ekf_lost_feature_ids.push_back(feature.id);
      }
    }


  }

  // Process ekf features that lose track
  rmLostFeaturesCov(ekf_lost_feature_ids);

  // Process useless nuisance state
  if (use_schmidt)
    rmUselessNuisanceState();

  // Update information in grid map
  updateGridMap();

  // Loop for features not in state
  for (auto iter = map_server.begin();
      iter != map_server.end(); ++iter) {
    // Rename the feature to be checked.
    auto& feature = iter->second;

    // TRUE if this feature is tracked now
    bool if_tracked_now =
          feature.observations.find(state_server.imu_state.id)!=feature.observations.end();

    if (!feature.in_state) {
      if (!if_tracked_now) {
        if (feature.observations.size() < least_Obs_Num) {
          invalid_feature_ids.push_back(feature.id);
          continue;
        }

        // Check if the feature can be initialized if it has not been.
        if (!feature.is_initialized) {
          if (!feature.checkMotion(state_server.imu_states_augment,if_tracked_now)) {
            invalid_feature_ids.push_back(feature.id); 
            continue;
          } else {
            if (!feature.initializePosition(state_server.imu_states_augment,state_server.imu_state.id)) { 
              invalid_feature_ids.push_back(feature.id);
              continue;
            }

          }
        }

        jacobian_row_size_msckf += 2*feature.observations.size() - 3;
        msckf_feature_ids.push_back(feature.id);
        msckf_lost_feature_ids.push_back(feature.id);
      } else {
        // TRUE if this feature is tracked now and have been tracked for too long
        bool if_tracked_long = feature.observations.size()>=max_track_len;

        // Pass the features that are still being tracked.
        if (!if_tracked_long) { 
          continue;
        }

        // Use this feature as ekf feature or msckf feature base on the occupation of map grid
        Vector2d xy =
          map_server[feature.id].observations[state_server.imu_state.id];
        int row = static_cast<int>((xy(1)-y_min)/grid_height);
        int col = static_cast<int>((xy(0)-x_min)/grid_width);
        int code = row*grid_cols + col;
        if (grid_map[code].size()<max_features && state_server.imu_state.time-last_ZUPT_time>5 &&  
            (state_server.feature_states.size()+ekf_new_feature_ids.size())<max_features*grid_rows*grid_cols) {
          // Intitialize it as ekf feature no matter if it hasn't been initialized as one.
          if (!feature.ekf_feature) {
            feature.is_initialized = false;
            if (feature.checkMotion(state_server.imu_states_augment,if_tracked_now))
              feature.initializeInvParamPosition(state_server.imu_states_augment,state_server.imu_state.id);
          }
          // Pass the feature that failed initialization
          if (!feature.is_initialized)
            continue;
          if (1==feature_idp_dim)
            jacobian_row_size_ekf_new += 2*(feature.observations.size()-1);   // observation of anchor frame is not used when 1d idp
          else
            jacobian_row_size_ekf_new += 2*feature.observations.size();
          ekf_new_feature_ids.push_back(feature.id);
          grid_map[code].push_back(feature.id);
        } else {
          // Try to initialize a msckf feature
          if (!feature.is_initialized) {
            if (feature.checkMotion(state_server.imu_states_augment,if_tracked_now))
              feature.initializePosition(state_server.imu_states_augment,state_server.imu_state.id);
          }
          if (!feature.is_initialized)
            continue;

          jacobian_row_size_msckf += 2*feature.observations.size() - 3;
          msckf_feature_ids.push_back(feature.id);
          msckf_lost_feature_ids.push_back(feature.id);
        }
      }
    }
  }

  // Remove the features that do not have enough measurements.
  for (const auto& feature_id : invalid_feature_ids) 
    map_server.erase(feature_id);

  // Return if there is no feature to be processed.
  if (msckf_feature_ids.size() == 0
      && ekf_new_feature_ids.size() == 0
      && ekf_feature_ids.size() == 0)
    return;

  // Do measurement update if ZUPT didn't happen.
  if ( !if_ZUPT ) {
    // Augment new ekf features into filter state
    for (const auto& feature_id : ekf_new_feature_ids) {
      map_server[feature_id].in_state = true;
      state_server.feature_states.push_back(feature_id);
    }

    // Stacked Jacobian and residuals for new EKF-SLAM features
    MatrixXd H_ekf_new = MatrixXd::Zero(jacobian_row_size_ekf_new,
          state_server.state_cov.cols()
          +feature_idp_dim*ekf_new_feature_ids.size());
    VectorXd r_ekf_new = VectorXd::Zero(jacobian_row_size_ekf_new);
    int stack_cntr = 0;
    vector<FeatureIDType> invalid_new_feature_ids(0);
    // Process new EKF-SLAM features
    for (const auto& feature_id : ekf_new_feature_ids) {
      auto& feature = map_server[feature_id];

      vector<StateIDType> state_ids(0);
      for (const auto& measurement : feature.observations)
        state_ids.push_back(measurement.first);

      MatrixXd H_xj;
      VectorXd r_j;
      featureJacobian_ekf_new(feature_id, state_ids, H_xj, r_j);

      // use msckf gating test to justify if this feature is useful
      MatrixXd H_xj_;
      VectorXd r_j_;
      featureJacobian_msckf(feature_id, state_ids, H_xj_, r_j_);

      if (gatingTestFeature(H_xj_, r_j_, 2*state_ids.size()-3)) {
        H_ekf_new.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
        r_ekf_new.segment(stack_cntr, r_j.rows()) = r_j;
        stack_cntr += H_xj.rows();
      } else {
        invalid_new_feature_ids.push_back(feature_id);
        if (1==feature_idp_dim)
          jacobian_row_size_ekf_new = jacobian_row_size_ekf_new-2*(feature.observations.size()-1);
        else
          jacobian_row_size_ekf_new = jacobian_row_size_ekf_new-2*feature.observations.size();
      }
    }
    // Resize H_ekf and r_ekf
    H_ekf_new.conservativeResize(stack_cntr, H_ekf_new.cols());
    r_ekf_new.conservativeResize(stack_cntr);
    // Delete new features which didn't pass gating test
    for (const auto& feature_id : invalid_new_feature_ids) {
      // Delete this feature, remove its relevant Jacobian;
      auto feature_itr = find(ekf_new_feature_ids.begin(),
            ekf_new_feature_ids.end(), feature_id);
      int feature_sequence = std::distance(
            ekf_new_feature_ids.begin(), feature_itr);
      int feature_state_start = state_server.state_cov.cols()+feature_idp_dim*feature_sequence;
      int feature_state_end = feature_state_start + feature_idp_dim;

      // Remove the corresponding rows and columns in Jacobian.
      if (feature_state_end < H_ekf_new.cols()) {
        H_ekf_new.block(0, feature_state_start,
            H_ekf_new.rows(), H_ekf_new.cols()-feature_state_end) =
          H_ekf_new.block(0, feature_state_end,
              H_ekf_new.rows(), H_ekf_new.cols()-feature_state_end);
      }
      H_ekf_new.conservativeResize(
          H_ekf_new.rows(), H_ekf_new.cols()-feature_idp_dim);

      // Remove this feature state from the state vector
      map_server[feature_id].in_state = false;
      // state_server.feature_states.erase(feature_itr);
      auto feature_itr_ = find(state_server.feature_states.begin(),
            state_server.feature_states.end(), feature_id);
      state_server.feature_states.erase(feature_itr_);

      // Remove this feature from ekf_new_feature_ids
      ekf_new_feature_ids.erase(feature_itr);
    }
    // Sparsify if needed
    if (ekf_new_feature_ids.size() > 0) {
      // Seperate two parts
      const MatrixXd& H_x_ekf_new = H_ekf_new.leftCols(
          H_ekf_new.cols()-feature_idp_dim*ekf_new_feature_ids.size());
      const MatrixXd& H_f_ekf_new = H_ekf_new.rightCols(
          feature_idp_dim*ekf_new_feature_ids.size());
      // Compute the left nullspace of H_f_ekf_new.
      JacobiSVD<MatrixXd> svd_helper(H_f_ekf_new, ComputeFullU | ComputeThinV);
      MatrixXd V = svd_helper.matrixU().rightCols(
              jacobian_row_size_ekf_new - feature_idp_dim*ekf_new_feature_ids.size());
      // Compute column space of H_f_ekf_new.
      SparseMatrix<double> H_sparse = H_f_ekf_new.sparseView();
      SPQR<SparseMatrix<double> > spqr_helper;
      spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
      spqr_helper.compute(H_sparse);
      MatrixXd Q;
      MatrixXd IdenMx = MatrixXd::Identity(jacobian_row_size_ekf_new,jacobian_row_size_ekf_new);
      (spqr_helper.matrixQ()*IdenMx).evalTo(Q);
      MatrixXd U = Q.leftCols(feature_idp_dim*ekf_new_feature_ids.size());
      // Project H_ekf_new and r_ekf_new into H_ekf_new_ and r_ekf_new_
      MatrixXd W = MatrixXd::Zero(jacobian_row_size_ekf_new, jacobian_row_size_ekf_new);
      W.leftCols(jacobian_row_size_ekf_new - feature_idp_dim*ekf_new_feature_ids.size()) = V;
      W.rightCols(feature_idp_dim*ekf_new_feature_ids.size()) = U;
      H_ekf_new = W.transpose() * H_ekf_new;
      r_ekf_new = W.transpose() * r_ekf_new;
    } else {
      H_ekf_new = MatrixXd::Zero(0,
        state_server.state_cov.cols()
        +feature_idp_dim*ekf_new_feature_ids.size());
      r_ekf_new = VectorXd::Zero(0);
    }

    // Stacked Jacobian and residuals for EKF-SLAM features
    MatrixXd H_ekf = MatrixXd::Zero(jacobian_row_size_ekf,
        state_server.state_cov.cols());
    VectorXd r_ekf = VectorXd::Zero(jacobian_row_size_ekf);
    stack_cntr = 0;
    // Process EKF-SLAM features
    for (const auto& feature_id : ekf_feature_ids) {
      MatrixXd H_xj;
      Vector2d r_j;

      featureJacobian_ekf(feature_id, H_xj, r_j);

      if (gatingTestFeature(H_xj, r_j, 2)) {
        H_ekf.block(stack_cntr, 0, H_xj.rows(), H_xj.cols()) = H_xj;
        r_ekf.segment(stack_cntr, r_j.rows()) = r_j;
        stack_cntr += H_xj.rows();
      }
    }
    // Resize H_ekf and r_ekf
    H_ekf.conservativeResize(stack_cntr, H_ekf.cols());
    r_ekf.conservativeResize(stack_cntr);
    // Decompose the final Jacobian matrix to reduce computational complexity as in Equation (28), (29).
    MatrixXd H_ekf_thin;
    VectorXd r_ekf_thin;
    if (H_ekf.rows() == 0 || r_ekf.rows() == 0 ||
        H_ekf.rows() <= H_ekf.cols()) {
      H_ekf_thin = H_ekf;
      r_ekf_thin = r_ekf;
    } else {
      // Convert H_ekf to a sparse matrix.
      SparseMatrix<double> H_sparse = H_ekf.sparseView();

      // Perform QR decompostion on H_sparse.
      SPQR<SparseMatrix<double> > spqr_helper;
      spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
      spqr_helper.compute(H_sparse);

      MatrixXd H_temp;
      VectorXd r_temp;
      (spqr_helper.matrixQ().transpose() * H_ekf).evalTo(H_temp);
      (spqr_helper.matrixQ().transpose() * r_ekf).evalTo(r_temp);

      H_ekf_thin = H_temp.topRows(state_server.state_cov.rows());
      r_ekf_thin = r_temp.head(state_server.state_cov.rows());
    }
    if (H_ekf_thin.rows()<H_ekf.rows())   // debug log
      printf("H_ekf.row = %ld, H_ekf_thin.rows = %ld",H_ekf.rows(),H_ekf_thin.rows());
    H_ekf = MatrixXd::Zero(H_ekf_thin.rows(),
        state_server.state_cov.cols());
    H_ekf.leftCols(H_ekf_thin.cols()) = H_ekf_thin;
    r_ekf = r_ekf_thin;

    // Stacked Jacobian and residuals for MSCKF features
    MatrixXd H_msckf = MatrixXd::Zero(jacobian_row_size_msckf,
        LEG_DIM+6*state_server.imu_states_augment.size());
    VectorXd r_msckf = VectorXd::Zero(jacobian_row_size_msckf);
    stack_cntr = 0;
    // Process the MSCKF features which lose track.
    for (const auto& feature_id : msckf_feature_ids) {   
      auto& feature = map_server[feature_id];

      vector<StateIDType> state_ids(0);
      for (const auto& measurement : feature.observations)
        state_ids.push_back(measurement.first);

      MatrixXd H_xj;
      VectorXd r_j;
      featureJacobian_msckf(feature_id, state_ids, H_xj, r_j); 

      if (gatingTestFeature(H_xj, r_j, 2*state_ids.size()-3)) { 
        H_msckf.block(stack_cntr, 0, H_xj.rows(), H_msckf.cols()) = H_xj.leftCols(H_msckf.cols());
        r_msckf.segment(stack_cntr, r_j.rows()) = r_j;
        stack_cntr += H_xj.rows();
      }
    }

    // check how many geometric features are used for update 
    // if (stack_cntr > 0)
    //   std::cout << "[Update state]: feature update using " << stack_cntr / 2 << " geometric features" << std::endl;

    // Resize H_msckf and r_msckf
    H_msckf.conservativeResize(stack_cntr, H_msckf.cols()); 
    r_msckf.conservativeResize(stack_cntr);   
    // Decompose the final Jacobian matrix to reduce computational
    // complexity as in Equation (28), (29).
    MatrixXd H_msckf_thin;
    VectorXd r_msckf_thin;
    if (H_msckf.rows() == 0 || r_msckf.rows() == 0 ||
        H_msckf.rows() <= H_msckf.cols()) {
      H_msckf_thin = H_msckf;
      r_msckf_thin = r_msckf;
    } else {
      // Convert H_msckf to a sparse matrix.
      SparseMatrix<double> H_sparse = H_msckf.sparseView();

      // Perform QR decompostion on H_sparse.
      SPQR<SparseMatrix<double> > spqr_helper;
      spqr_helper.setSPQROrdering(SPQR_ORDERING_NATURAL);
      spqr_helper.compute(H_sparse);

      MatrixXd H_temp;
      VectorXd r_temp;
      (spqr_helper.matrixQ().transpose() * H_msckf).evalTo(H_temp);
      (spqr_helper.matrixQ().transpose() * r_msckf).evalTo(r_temp);

      H_msckf_thin = H_temp.topRows(LEG_DIM+state_server.imu_states_augment.size()*6);  
      r_msckf_thin = r_temp.head(LEG_DIM+state_server.imu_states_augment.size()*6);   
    }
    H_msckf = MatrixXd::Zero(H_msckf_thin.rows(),
        state_server.state_cov.cols());
    H_msckf.leftCols(H_msckf_thin.cols()) = H_msckf_thin;
    r_msckf = r_msckf_thin;

    // Measurement update.
    measurementUpdate_hybrid(
      H_ekf_new, r_ekf_new, H_ekf, r_ekf, H_msckf, r_msckf);

    // Delete redundant features in grid map
    // delRedundantFeatures();
  } else {
    for (int i = 0; i < msckf_feature_ids.size(); i++) {
      auto feature_id = msckf_feature_ids[i];
      map_server[feature_id].is_initialized = false;
    }
  }

  // Remove all processed and untracked features from the map.
  for (int i = 0; i < msckf_lost_feature_ids.size(); i++) { 
    auto feature_id = msckf_lost_feature_ids[i];
    // if (!processed_but_tracked[i])
      map_server.erase(feature_id);
  }

  return;
}


void OrcVIO::findRedundantImuStates(
    vector<StateIDType>& rm_state_ids) {

  // Move the iterator to the key position.
  auto key_state_iter = state_server.imu_states_augment.end();
  for (int i = 0; i < 4; ++i)
    --key_state_iter;
  auto state_iter = key_state_iter;
  ++state_iter;
  auto first_state_iter = state_server.imu_states_augment.begin();

  // Pose of the key camera state.
  const Vector3d key_position =
    key_state_iter->second.position_cam;
  const Matrix3d key_rotation = key_state_iter->second.orientation_cam;

  // Mark the camera states to be removed based on the
  // motion between states.
  for (int i = 0; i < 2; ++i) {
    const Vector3d position =
      state_iter->second.position_cam;
    const Matrix3d rotation = state_iter->second.orientation_cam.transpose();

    double distance = (position-key_position).norm();
    double angle = AngleAxisd(
        rotation*key_rotation).angle();

    if (angle < rotation_threshold &&
        distance < translation_threshold &&
        tracking_rate > tracking_rate_threshold) { 
      rm_state_ids.push_back(state_iter->first);
      ++state_iter;
    } else {
      rm_state_ids.push_back(first_state_iter->first);
      ++first_state_iter;
      --state_iter;
      --state_iter;
    }
  }

  // Sort the elements in the output vector.
  sort(rm_state_ids.begin(), rm_state_ids.end());

  return;
}


void OrcVIO::pruneImuStateBuffer() {

  vector<StateIDType> rm_imu_state_ids(0);
  vector<StateIDType> new_nui_state_ids(0);

  if ( !if_ZUPT ) {
    if (state_server.imu_states_augment.size() < sw_size) 
      return;

    // Find two camera states to be removed.
    findRedundantImuStates(rm_imu_state_ids); 

  } else {
    // previous state must be deleted if apply ZUPT,
    // no matter whether the cam size is out of range.
    rm_imu_state_ids.push_back(state_server.imu_state.id-1);
  }

  // Update information for every related feature.
  // And find the size of the Jacobian matrix.
  int jacobian_row_size = 0;
  vector<FeatureIDType> used_IDs(0);
  for (auto& item : map_server) {
    auto& feature = item.second;

    // Check how many camera states to be removed are associated
    // with this feature.
    vector<StateIDType> involved_state_ids(0);
    for (const auto& state_id : rm_imu_state_ids) { 
      if (feature.observations.find(state_id) !=
          feature.observations.end())
        involved_state_ids.push_back(state_id);
    }

    if (involved_state_ids.size() == 0) continue;

    if (feature.in_state) {
      // Change anchor camera of a feature if the current one is to be deleted
      vector<StateIDType>::iterator it = find(
        involved_state_ids.begin(), involved_state_ids.end(), feature.id_anchor);
      if (it != involved_state_ids.end()) {
        // if (use_schmidt) {
        if (use_schmidt &&
            state_server.imu_state.id-feature.id_anchor>2) {  // Only use mature pose as nuisance state, the threshold '2' is decide by criterion in findRedundantImuStates
          // Save observation information for new nuisance state
          state_server.nui_features[feature.id_anchor].push_back(feature.id);
          // Prepare the id of new nuisance state
          if (find(new_nui_state_ids.begin(), new_nui_state_ids.end(), feature.id_anchor)
              == new_nui_state_ids.end())
            new_nui_state_ids.push_back(feature.id_anchor);
        } else {
          StateIDType new_id;
          if (3==feature_idp_dim) {
            // New anchor ID.
            new_id = state_server.imu_state.id;
            // New anchor pose.
            const IMUState_Aug& imu_state_new = state_server.imu_states_augment[new_id];
            Matrix3d R_c2w_new = imu_state_new.orientation_cam;
            const Vector3d& t_c_w_new = imu_state_new.position_cam;
            // Compute new inverse depth feature position.
            const Vector3d& p_w = feature.position;
            Vector3d p_new = R_c2w_new.inverse() * (p_w-t_c_w_new);
            // Compute new inverse depth feature position.
            feature.invParam(0) = p_new(0)/p_new(2);
            feature.invParam(1) = p_new(1)/p_new(2);
            feature.invParam(2) = 1/p_new(2);
            updateFeatureCov_3didp(feature.id,
              feature.id_anchor, new_id);
          } else {
            // New anchor ID.
            new_id = getNewAnchorId(feature, involved_state_ids);
            // new_id = state_server.imu_state.id;
            // New anchor pose.
            const IMUState_Aug& imu_state_new = state_server.imu_states_augment[new_id];
            Matrix3d R_c2w_new = imu_state_new.orientation_cam;
            const Vector3d& t_c_w_new = imu_state_new.position_cam;
            // Compute new inverse depth feature position.
            const Vector3d& p_w = feature.position;
            Vector3d p_new = R_c2w_new.inverse() * (p_w-t_c_w_new);
            // Compute new inverse depth and correct observation for new anchor frame.
            feature.invDepth = 1/p_new(2);
            // Fix observation if the correction is small
            Vector2d obs_fix = Vector2d(p_new(0)/p_new(2), p_new(1)/p_new(2));
            feature.obs_anchor(0) = obs_fix(0);
            feature.obs_anchor(1) = obs_fix(1);

            updateFeatureCov_1didp(feature.id,
              feature.id_anchor, new_id);
          }

          // Update anchor id.
          feature.id_anchor = new_id;
        }
      }
    } else {
      if (feature.is_initialized) { 
        vector<StateIDType>::iterator it = find(
          involved_state_ids.begin(), involved_state_ids.end(), feature.id_anchor);
        if (it != involved_state_ids.end()) {
          if (0) {    // NOTE: Potential EKF features does not produce nuisance state
          // if (use_schmidt && feature.ekf_feature) {
          //   // Save observation information for new nuisance state
          //   state_server.nui_features[feature.id_anchor].push_back(feature.id);
          //   // Prepare the id of new nuisance state
          //   if (find(new_nui_state_ids.begin(), new_nui_state_ids.end(), feature.id_anchor)
          //     == new_nui_state_ids.end()) {
          //     new_nui_state_ids.push_back(feature.id_anchor);
          //   }
          } else {
            StateIDType new_id;
            if (3==feature_idp_dim) {
              // New anchor ID.
              new_id = state_server.imu_state.id;
              // New anchor pose.
              const IMUState_Aug& imu_state_new = state_server.imu_states_augment[new_id];
              Matrix3d R_c2w_new = imu_state_new.orientation_cam;
              const Vector3d& t_c_w_new = imu_state_new.position_cam;
              // Compute new inverse depth feature position.
              const Vector3d& p_w = feature.position;
              Vector3d p_new = R_c2w_new.inverse() * (p_w-t_c_w_new);
              // Compute new inverse depth feature position.
              feature.invParam(0) = p_new(0)/p_new(2);
              feature.invParam(1) = p_new(1)/p_new(2);
              feature.invParam(2) = 1/p_new(2);
            } else {
              // New anchor ID.
              new_id = getNewAnchorId(feature, involved_state_ids);
              // new_id = state_server.imu_state.id;
              // New anchor pose.
              const IMUState_Aug& imu_state_new = state_server.imu_states_augment[new_id];
              Matrix3d R_c2w_new = imu_state_new.orientation_cam;
              const Vector3d& t_c_w_new = imu_state_new.position_cam;
              // Compute new inverse depth feature position.
              const Vector3d& p_w = feature.position;
              Vector3d p_new = R_c2w_new.inverse() * (p_w-t_c_w_new);
              // Compute new inverse depth and correct observation for new anchor frame.
              feature.invDepth = 1/p_new(2);
              feature.obs_anchor(0) = feature.observations[new_id](0);   
              feature.obs_anchor(1) = feature.observations[new_id](1);
            }

            // Update anchor id.
            feature.id_anchor = new_id;
          }
        }
      }

      // Try to use feature as msckf feature, do not use potential ekf features
      if (!if_ZUPT && !feature.ekf_feature &&
          involved_state_ids.size()>1) {

        bool if_tracked = feature.observations.find(state_server.imu_state.id)
                          !=feature.observations.end();
        if (!feature.is_initialized) {     
          // Check if the feature can be initialize.
          if (!feature.checkMotion(state_server.imu_states_augment,if_tracked)) {
            continue;
          } else {
            if (!feature.initializePosition_AssignAnchor(state_server.imu_states_augment)) {
              continue;
            }
          }
        }
        used_IDs.push_back(feature.id);
        jacobian_row_size += 2*involved_state_ids.size() - 3;
      }

      // debug log
      if (feature.observations.find(state_server.imu_state.id) ==
          feature.observations.end())
          printf("A LOST FEATURE SHOULD NOT BE HERE !\n");
    }
  }

  // Apply MSCKF measurement update if ZUPT didn't happen
  if (!if_ZUPT && used_IDs.size()!=0) {
    // Compute the Jacobian and residual.
    MatrixXd H_x = MatrixXd::Zero(jacobian_row_size,
          state_server.state_cov.cols());
    VectorXd r = VectorXd::Zero(jacobian_row_size);
    int stack_cntr = 0;
    for (auto& item : map_server) {
      auto& feature = item.second;

      // Check how many camera states to be removed are associated
      // with this feature.
      vector<StateIDType> involved_state_ids(0);
      for (const auto& state_id : rm_imu_state_ids) {   
        if (feature.observations.find(state_id) !=    
            feature.observations.end())
          involved_state_ids.push_back(state_id);
      }

      // Not using ekf features
      if (find(used_IDs.begin(),used_IDs.end(),feature.id)
          !=used_IDs.end()) {
        // debug log
        if (involved_state_ids.size() == 0)
          printf("Size of involved_state_ids should not be 0 !!");
        if (involved_state_ids.size() == 1)
          printf("Size of involved_state_ids should not be 1 !!");

        MatrixXd H_xj;
        VectorXd r_j;
        featureJacobian_msckf(feature.id, involved_state_ids, H_xj, r_j); 

        if (gatingTestFeature(H_xj, r_j, 2*involved_state_ids.size()-3)) {   
          H_x.block(stack_cntr, 0, H_xj.rows(), H_x.cols()) = H_xj;
          r.segment(stack_cntr, r_j.rows()) = r_j;
          stack_cntr += H_xj.rows();
        }
      }

      for (const auto& state_id : involved_state_ids) { 
          feature.observations.erase(state_id);
      }
    }

    H_x.conservativeResize(stack_cntr, H_x.cols());
    r.conservativeResize(stack_cntr); 

    // Perform measurement update.
    measurementUpdate_msckf(H_x, r); 
  } else {
    // Simply delete observations if ZUPT happened or no msckf features.
    for (auto& item : map_server) {
      auto& feature = item.second;

      // Check how many camera states to be removed are associated
      // with this feature.
      vector<StateIDType> involved_state_ids(0);
      for (const auto& state_id : rm_imu_state_ids) { 
        if (feature.observations.find(state_id) !=  
            feature.observations.end())
          involved_state_ids.push_back(state_id);
      }

      if (involved_state_ids.size() == 0) continue;

      for (const auto& state_id : involved_state_ids) {  
          feature.observations.erase(state_id);
      }
    }
  }

  // Update state covariance
  for (const auto& state_id : rm_imu_state_ids) {
    int state_sequence = std::distance(state_server.imu_states_augment.begin(),
        state_server.imu_states_augment.find(state_id));
    int state_start = LEG_DIM + 6*state_sequence;
    int state_end = state_start + 6;

    if (use_schmidt && find(new_nui_state_ids.begin(), new_nui_state_ids.end(), state_id)
          != new_nui_state_ids.end()) {
      // Reconstruct the covariance matrix.
      if (state_end < state_server.state_cov.rows()) {
        Matrix<double, 6, 6> P_ss =
              state_server.state_cov.block<6, 6>(state_start, state_start);
        MatrixXd P_os_1 = state_server.state_cov.block(
              state_start, 0, 6, state_start);
        MatrixXd P_os_2 = state_server.state_cov.block(
              state_start, state_end,
              6, state_server.state_cov.cols()-state_end);

        state_server.state_cov.block(state_start, 0,
            state_server.state_cov.rows()-state_end,
            state_server.state_cov.cols()) =
          state_server.state_cov.block(state_end, 0,
              state_server.state_cov.rows()-state_end,
              state_server.state_cov.cols());

        state_server.state_cov.block(0, state_start,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-state_end) =
          state_server.state_cov.block(0, state_end,
              state_server.state_cov.rows(),
              state_server.state_cov.cols()-state_end);

        state_server.state_cov.block<6, 6>(
            state_server.state_cov.rows()-6,
            state_server.state_cov.cols()-6) = P_ss;
        state_server.state_cov.block(
            state_server.state_cov.rows()-6, 0,
            6, state_start) = P_os_1;
        state_server.state_cov.block(
            state_server.state_cov.rows()-6, state_start,
            6, state_server.state_cov.cols()-state_end) = P_os_2;
        state_server.state_cov.block(
            0, state_server.state_cov.cols()-6,
            state_start, 6) = P_os_1.transpose();
        state_server.state_cov.block(
            state_start, state_server.state_cov.cols()-6,
            state_server.state_cov.rows()-state_end, 6) = P_os_2.transpose();
      }
      // Save nuisance imu/cam state
      state_server.nui_ids.push_back(state_id);
      state_server.nui_imu_states[state_id] = state_server.imu_states_augment[state_id];
    } else {
      // Remove the corresponding rows and columns in the state
      // covariance matrix.
      if (state_end < state_server.state_cov.rows()) {
        state_server.state_cov.block(state_start, 0,
            state_server.state_cov.rows()-state_end,
            state_server.state_cov.cols()) =
          state_server.state_cov.block(state_end, 0,
              state_server.state_cov.rows()-state_end,
              state_server.state_cov.cols());

        state_server.state_cov.block(0, state_start,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-state_end) =
          state_server.state_cov.block(0, state_end,
              state_server.state_cov.rows(),
              state_server.state_cov.cols()-state_end);
      }
      state_server.state_cov.conservativeResize(
          state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    }

    // remove the timestamp from the current window 
    double time_to_erase = state_server.imu_states_augment[state_id].time;
    cur_window_timestamps.erase(std::remove(cur_window_timestamps.begin(), cur_window_timestamps.end(), time_to_erase), 
      cur_window_timestamps.end());

    // Remove this imu state in the state vector.
    state_server.imu_states_augment.erase(state_id);

  }

  return;
}


// Get T_b_w
Eigen::Isometry3d OrcVIO::getTbw() {
  // Convert the IMU frame to the body frame.
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
  T_i_w.linear() = imu_state.orientation;

  T_i_w.translation() = imu_state.position;

  Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
          IMUState::T_imu_body.inverse();    

  return T_b_w;
}

// Get T_c_w
Eigen::Isometry3d OrcVIO::getTcw() {
  // Convert the IMU frame to the body frame.
  const IMUState_Aug& imu_state = state_server.imu_states_augment[state_server.imu_state.id];
  Eigen::Isometry3d T_c_w = Eigen::Isometry3d::Identity();
  
  T_c_w.linear() = imu_state.orientation_cam;
  T_c_w.translation() = imu_state.position_cam;

  return T_c_w;
}

// Get velocity
Eigen::Vector3d OrcVIO::getVel() {
  const IMUState& imu_state = state_server.imu_state;
  Eigen::Vector3d body_velocity =
          IMUState::T_imu_body.linear() * imu_state.velocity;

  return body_velocity;
}


// Get P_pose
Eigen::Matrix<double, 6, 6> OrcVIO::getPpose() {
  // Convert the covariance.
  Matrix3d P_oo = state_server.state_cov.block<3, 3>(0, 0);   
  Matrix3d P_op = state_server.state_cov.block<3, 3>(0, 6);
  Matrix3d P_po = state_server.state_cov.block<3, 3>(6, 0);
  Matrix3d P_pp = state_server.state_cov.block<3, 3>(6, 6);
  Matrix<double, 6, 6> P_imu_pose = Matrix<double, 6, 6>::Zero();
  P_imu_pose << P_pp, P_po, P_op, P_oo;

  Matrix<double, 6, 6> H_pose = Matrix<double, 6, 6>::Zero();
  H_pose.block<3, 3>(0, 0) = IMUState::T_imu_body.linear();
  H_pose.block<3, 3>(3, 3) = IMUState::T_imu_body.linear();
  Matrix<double, 6, 6> P_body_pose = H_pose * P_imu_pose * H_pose.transpose();

  return P_body_pose;
}


// Get P_vel
Eigen::Matrix3d OrcVIO::getPvel() {
  // Construct the covariance for the velocity.
  Matrix3d P_imu_vel = state_server.state_cov.block<3, 3>(3, 3);  
  Matrix3d H_vel = IMUState::T_imu_body.linear();
  Matrix3d P_body_vel = H_vel * P_imu_vel * H_vel.transpose();

  return P_body_vel;
}


// Get poses of augmented IMU states
void OrcVIO::getSwPoses(vector<Eigen::Isometry3d>& swPoses) {
  swPoses.clear();
  for (auto itr : state_server.imu_states_augment) {
    const auto& imu_state = itr.second;
    Eigen::Isometry3d T_i_w = Eigen::Isometry3d::Identity();
    T_i_w.linear() = imu_state.orientation;

    T_i_w.translation() = imu_state.position;
    Eigen::Isometry3d T_b_w = IMUState::T_imu_body * T_i_w *
            IMUState::T_imu_body.inverse(); 
    swPoses.push_back(T_b_w);
  }
}


// Get position of stable map points
void OrcVIO::getStableMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints) {
  for (auto item : lost_slam_features)
    mMapPoints[item.first] = item.second.position;
  lost_slam_features.clear();
}


void OrcVIO::getActiveMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints) {
  for (auto item : active_slam_features)
    mMapPoints[item.first] = item.second.position;
  active_slam_features.clear();
}

void OrcVIO::getMSCKFMapPointPositions(std::map<orcvio::FeatureIDType,Eigen::Vector3d>& mMapPoints) {
  for (auto item : map_server)
    mMapPoints[item.first] = item.second.position;
}

//  reset First Estimate Point to current estimate
void OrcVIO::resetFejPoint () {
  // reset FEJ of current imu state
  state_server.imu_state_FEJ_now = state_server.imu_state;

  // reset FEJ of historical imu state
  for (auto iter = state_server.imu_states_augment.begin();
        iter != state_server.imu_states_augment.end();
        ++iter) {
    iter->second.position_FEJ = iter->second.position;
  }

  // debug log
  printf("RESET FIRST ESTIMATE POINT AT %fs!",state_server.imu_state.time-take_off_stamp);
}

// check if need ZUPT, 
bool OrcVIO::checkZUPTFeat () {
  if ( coarse_feature_dis.empty()
      || coarse_feature_dis.size()<20 ) {  
    list<double>().swap(coarse_feature_dis);
    return false;
  }

  // ignore outliers rudely
  coarse_feature_dis.sort();
  auto itr = coarse_feature_dis.end();
  for (int i = 0; i < 9; i++)  
    itr--;
  double maxDis = *itr;
  // double minDis = *(coarse_feature_dis.begin());

  // have to be cleared
  list<double>().swap(coarse_feature_dis);

  // for debugging 
  // std::cout << "maxDis " << maxDis << std::endl; 

  if ( maxDis<zupt_max_feature_dis ) {
    // Delete all features from state
    if (state_server.feature_states.size()>0) {
      state_server.state_cov.conservativeResize(
            state_server.state_cov.rows()-feature_idp_dim*state_server.feature_states.size(),
            state_server.state_cov.cols()-feature_idp_dim*state_server.feature_states.size());
      for (auto id : state_server.feature_states) {
        auto& feature = map_server[id];
        feature.is_initialized = false;
        feature.ekf_feature = false;
        feature.in_state = false;
      }
      state_server.feature_states.clear();
    }
    // Do ZUPT
    measurementUpdate_ZUPT_vpq();
    printf("[ZUPT]: Zero velocity update\n");
    return true;
  } else {
    // printf("[ZUPT]: No zero velocity update\n");
    return false;
  }

}

// check if need ZUPT using IMU data 
// ref https://docs.openvins.com/update-zerovelocity.html
bool OrcVIO::checkZUPTIMU () {

  bool do_zupt = false;

  // Check that we have at least one measurement to propagate with
  if(imu_recent_zupt.empty() || imu_recent_zupt.size() < 2) {
      printf("[ZUPT]: There is no IMU data to check for zero velocity with!!\n");
      return do_zupt;
  }

  /// Gyroscope white noise covariance
  double sigma_w_2 = pow(1.6968e-04, 2);
  /// Accelerometer white noise covariance
  double sigma_a_2 = pow(2.0000e-3, 2);
  /// Gyroscope random walk (rad/s^2/sqrt(hz))
  double sigma_wb = 1.9393e-05;
  /// Accelerometer random walk (m/s^3/sqrt(hz))
  double sigma_ab = 3.0000e-03;
  /// What chi-squared multipler we should apply
  int chi2_multipler = 1;
  /// Max velocity (m/s) that we should consider a zupt with
  double _zupt_max_velocity = 0.25;

  // Loop through all our IMU and construct the residual and Jacobian
  // State order is: [q_ItoG, bg, ba]
  // Measurement order is: [w_true = 0, a_true = 0 or v_k+1 = 0]
  
  double dt_summed = 0;
  double dt = 0.0;

  // int max_imu_size = min(10, (int)imu_recent_zupt.size()-1);
  int max_imu_size = (int)imu_recent_zupt.size()-1;

  int h_size = 9;
  int m_size = 6 * max_imu_size;

  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(m_size, h_size);
  Eigen::VectorXd res = Eigen::VectorXd::Zero(m_size);
  Eigen::MatrixXd R = Eigen::MatrixXd::Identity(m_size, m_size);

  Eigen::Matrix3d wRi = state_server.imu_state.orientation;

  for(size_t i=0; i < max_imu_size; i++) {

    // Precomputed values
    dt = imu_recent_zupt.at(i+1).timeStampToSec - imu_recent_zupt.at(i).timeStampToSec;

    // Convert the msgs.
    Vector3d m_gyro = imu_recent_zupt.at(i).angular_velocity;
    Vector3d m_acc = imu_recent_zupt.at(i).linear_acceleration;

    // Remove the bias from the measured gyro and acceleration
    IMUState& imu_state = state_server.imu_state;
    Vector3d f = m_acc-imu_state.acc_bias;  
    // unbiased acc  
    Vector3d acc = state_server.Ma*f;

    Vector3d w = m_gyro-state_server.As*acc-imu_state.gyro_bias;
    // unbiased gyro 
    Vector3d gyro = state_server.Tg*w;

    // Measurement residual (true value is zero)
    res.block(6*i+0,0,3,1) = -gyro;
    
    // ignore residual in gyro 
    res.block(6*i+0,0,3,1) *= 0;

    res.block(6*i+3,0,3,1) = -wRi*acc - IMUState::gravity;

    // Measurement Jacobian    
    if (use_left_perturbation_flag)
    {
      // using left perturbation 
      H.block(6*i+0,3,3,3) = Eigen::Matrix<double,3,3>::Identity();

      H.block(6*i+3,0,3,3) = skewSymmetric(wRi * acc);
      H.block(6*i+3,6,3,3) = wRi;
    }
    else 
    {
      // using right perturbation 
      H.block(6*i+0,3,3,3) = Eigen::Matrix<double,3,3>::Identity();

      H.block(6*i+3,0,3,3) = wRi * skewSymmetric(acc);
      H.block(6*i+3,6,3,3) = wRi; 
    }

    // Measurement noise (convert from continuous to discrete)
    // Note the dt time might be different if we have "cut" any imu measurements
    R.block(6*i+0,6*i+0,3,3) *= sigma_w_2/dt;
    R.block(6*i+3,6*i+3,3,3) *= sigma_a_2/dt;

    dt_summed += dt;

  }

  // Next propagate the biases forward in time
  // NOTE: G*Qd*G^t = dt*Qd*dt = dt*Qc
  Eigen::MatrixXd Q_bias = Eigen::MatrixXd::Identity(6,6);
  Q_bias.block(0,0,3,3) *= dt_summed * sigma_wb;
  Q_bias.block(3,3,3,3) *= dt_summed * sigma_ab;

  // Chi2 distance check
  // NOTE: we also append the propagation we "would do before the update" if this was to be accepted
  // NOTE: we don't propagate first since if we fail the chi2 then we just want to return and do normal logic
  Eigen::MatrixXd P = state_server.state_cov;
  Eigen::MatrixXd P_marg = Eigen::MatrixXd::Identity(h_size, h_size);

  // diagnoal terms 
  // theta theta 
  P_marg.block(0,0,3,3) = P.block(0,0,3,3);
  // bg bg 
  P_marg.block(3,3,3,3) = P.block(9,9,3,3);
  // ba ba 
  P_marg.block(6,6,3,3) = P.block(12,12,3,3);

  // cross correlation 
  // theta bg 
  P_marg.block(0,3,3,3) = P.block(0,9,3,3);
  P_marg.block(3,0,3,3) = P.block(9,0,3,3);
  // theta ba
  P_marg.block(0,6,3,3) = P.block(0,12,3,3);
  P_marg.block(6,0,3,3) = P.block(12,0,3,3);
  // bg ba 
  P_marg.block(3,6,3,3) = P.block(9,12,3,3);
  P_marg.block(6,3,3,3) = P.block(12,9,3,3);
  
  P_marg.block(3,3,6,6) += Q_bias;

  double chi2;
  // this works in both debug and release mode 
  // if there's too many IMU measurements, turn this off 
  // otherwise it's too slow and nodelet will freeze 
  if (H.rows() < 1.5 * chi_squre_num_threshold)
  {
    Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
    chi2 = res.dot(S.llt().solve(res));
  }
  else 
  {
    chi2 = INT_MAX;
  }
  // this only works in release mode, 
  // in debug mode, the IMU callback will lead to large size H 
  Eigen::MatrixXd S = H * P_marg * H.transpose() + R;
  chi2 = res.dot(S.llt().solve(res));

  // Get our threshold (we precompute up to chi_squre_num_threshold but handle the case that it is more)
  double chi2_check;
  if(res.rows() < chi_squre_num_threshold) {
      chi2_check = chi_squared_table_zupt[res.rows()];
  } else {
      boost::math::chi_squared chi_squared_dist(res.rows());
      chi2_check = boost::math::quantile(chi_squared_dist, chi_square_threshold_zupt);
      // for debugging
      // printf("[ZUPT]: chi2_check over the residual limit - %d\n", (int)res.rows());
  }

  // Check if we are currently zero velocity
  // We need to pass the chi2 and not be above our velocity threshold
  Eigen::Vector3d cur_velocity = getVel();

  if(chi2 > chi2_multipler * chi2_check || cur_velocity.norm() > _zupt_max_velocity) {
      // for debugging 
      // printf("[ZUPT]: rejected zero velocity |v_IinG| = %.3f (chi2 %.3f - %.3f)\n", cur_velocity.norm(),chi2,chi2_multipler*chi2_check);
      do_zupt = false;
  }
  else 
  {
    // printf("[ZUPT]: accepted zero velocity |v_IinG| = %.3f (chi2 %.3f < %.3f)\n", cur_velocity.norm(),chi2,chi2_multipler*chi2_check);
    do_zupt = true;
  }

  if ( do_zupt ) {
    // Delete all features from state
    if (state_server.feature_states.size()>0) {
      state_server.state_cov.conservativeResize(
            state_server.state_cov.rows()-feature_idp_dim*state_server.feature_states.size(),
            state_server.state_cov.cols()-feature_idp_dim*state_server.feature_states.size());
      for (auto id : state_server.feature_states) {
        auto& feature = map_server[id];
        feature.is_initialized = false;
        feature.ekf_feature = false;
        feature.in_state = false;
      }
      state_server.feature_states.clear();
    }
    // Do ZUPT
    measurementUpdate_ZUPT_vpq();
    // printf("[ZUPT]: Zero velocity update\n");
    return true;
  } else
    return false;

}


void OrcVIO::measurementUpdate_ZUPT_vpq () {
  // ZUPT measurement Jacobian
  const int N = state_server.imu_states_augment.size();
  MatrixXd H = MatrixXd::Zero(9, state_server.state_cov.cols());
  H.block<3, 3>(0, 3) = Matrix3d::Identity();                   // zupt_v current
  H.block<3, 3>(3, LEG_DIM+6*N-3) = Matrix3d::Identity();        // zupt_p current
  H.block<3, 3>(3, LEG_DIM+6*N-9) = -Matrix3d::Identity();       // zupt_p previous
  H.block<3, 3>(6, LEG_DIM+6*N-6) = -0.5*Matrix3d::Identity();   // zupt_q current
  H.block<3, 3>(6, LEG_DIM+6*N-12) = 0.5*Matrix3d::Identity();   // zupt_q previous

  // ZUPT residual
  VectorXd r = VectorXd::Zero(9);
  // r_v
  r.segment<3>(0) = -state_server.imu_state.velocity;

  // // ignore update using zero velocity  
  // r.segment<3>(0) = -state_server.imu_state.velocity * 0;

  // r_p
  const Vector3d& p_curr = state_server.imu_states_augment[
    state_server.imu_state.id].position;
  const Vector3d& p_prev = state_server.imu_states_augment[
    state_server.imu_state.id-1].position;
  r.segment<3>(3) = -(p_curr-p_prev);

  // // ignore update using identical position  
  // r.segment<3>(3) = -(p_curr-p_prev) * 0;

  // r_q
  // const Vector4d& q_c = state_server.imu_states_augment[
  //   state_server.imu_state.id].orientation;
  const Vector4d& q_c = rotationToQuaternion(state_server.imu_states_augment[
    state_server.imu_state.id].orientation);
  
  Quaterniond q_curr(q_c(3), q_c(0), q_c(1), q_c(2));
  // const Vector4d& q_p = state_server.imu_states_augment[
  //   state_server.imu_state.id-1].orientation;
  const Vector4d& q_p = rotationToQuaternion(state_server.imu_states_augment[
    state_server.imu_state.id-1].orientation);

  Quaterniond q_prev(q_p(3), q_p(0), q_p(1), q_p(2));
  Quaterniond dq = q_curr*q_prev.conjugate();
  r.segment<3>(6) = Vector3d(dq.x(), dq.y(), dq.z());

  // // ignore update using identical rotation 
  // r.segment<3>(6) = Vector3d(0, 0, 0);

  // Construct ZUPT measurement corvriance
  MatrixXd R_ZUPT = MatrixXd::Zero(9, 9);
  R_ZUPT.block<3, 3>(0, 0) = zupt_noise_v*Matrix3d::Identity();
  R_ZUPT.block<3, 3>(3, 3) = zupt_noise_p*Matrix3d::Identity();
  R_ZUPT.block<3, 3>(6, 6) = zupt_noise_q*Matrix3d::Identity();

  // Compute the Kalman gain.
  MatrixXd& P = state_server.state_cov;
  MatrixXd S = H*P*H.transpose() + R_ZUPT;
  MatrixXd K_transpose = S.ldlt().solve(H*P); 
  MatrixXd K = K_transpose.transpose();

  // Compute the error of the state.
  VectorXd delta_x = K * r;

  // increment IMU and camera state 
  incrementState_IMUCam(delta_x);

  // Update the augmented feature states
  int base_cntr = LEG_DIM+6*state_server.imu_states_augment.size();
  auto feature_itr = state_server.feature_states.begin();
  for (int i = 0; i < state_server.feature_states.size();
    ++i, feature_itr++) {
    FeatureIDType feature_id = (*feature_itr);
    IMUState_Aug imu_state_aug;
    if (use_schmidt && find(state_server.nui_ids.begin(), state_server.nui_ids.end(),
        map_server[feature_id].id_anchor)!=state_server.nui_ids.end())
      imu_state_aug = state_server.nui_imu_states[map_server[feature_id].id_anchor];
    else if (state_server.imu_states_augment.find(map_server[feature_id].id_anchor)
        != state_server.imu_states_augment.end())
      imu_state_aug = state_server.imu_states_augment[map_server[feature_id].id_anchor];
    else
      printf("ERROR HAPPENED IN VPQ");
    Matrix3d R_c2w = imu_state_aug.orientation_cam;
    const Vector3d& t_c_w = imu_state_aug.position_cam;
    Vector3d p_c;
    if (3==feature_idp_dim) {
      // Update invParam
      const Vector3d& delta_f_aug = delta_x.segment<3>(base_cntr+i*3);
      map_server[feature_id].invParam += delta_f_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].invParam(0)/map_server[feature_id].invParam(2);
      p_c(1) = map_server[feature_id].invParam(1)/map_server[feature_id].invParam(2);
      p_c(2) = 1/map_server[feature_id].invParam(2);
    } else {
      // Update invDepth
      double delta_rho_aug = delta_x(base_cntr+i);
      map_server[feature_id].invDepth += delta_rho_aug;
      // Update position in world frame
      p_c(0) = map_server[feature_id].obs_anchor(0)/map_server[feature_id].invDepth;
      p_c(1) = map_server[feature_id].obs_anchor(1)/map_server[feature_id].invDepth;
      p_c(2) = 1/map_server[feature_id].invDepth;
    }
    Vector3d p_w = R_c2w*p_c + t_c_w;
    map_server[feature_id].position = p_w;
  }

  // Update state covariance.
  MatrixXd I_KH = MatrixXd::Identity(K.rows(), H.cols()) - K*H;  
  if (use_schmidt && state_server.nui_ids.size()>0) {
    MatrixXd P_nui = P.block(
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size());
    P = I_KH*P;
    P.block(
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      base_cntr+feature_idp_dim*state_server.feature_states.size(),
      6*state_server.nui_ids.size(), 6*state_server.nui_ids.size()) = P_nui;
  } else {
    P = I_KH*P;    
  }

  // Fix the covariance to be symmetric
  P = ((P + P.transpose()) / 2.0).eval();

  last_update_time = state_server.imu_state.time;

  last_ZUPT_time = state_server.imu_state.time;

  return;
}


void OrcVIO::updateFeatureCov_3didp(const FeatureIDType& feature_id,
        const StateIDType& old_state_id, const StateIDType& new_state_id) {
  const size_t state_size = state_server.imu_states_augment.size();
  const size_t feature_size = state_server.feature_states.size();

  const Feature& feature = map_server[feature_id];

  // Feature position in world frame.
  const Vector3d& p_w = feature.position;

  // Imu-camera extrinsic.
  const Matrix3d& R_b2c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_b = state_server.imu_state.t_cam0_imu;

  // Poses of old camera and corresponding imu.
  const IMUState_Aug& imu_state_old = state_server.imu_states_augment[old_state_id];
  Matrix3d R_b2w_old = imu_state_old.orientation;

  const Vector3d& t_b_w_old = imu_state_old.position;
  Matrix3d R_c2w_old = imu_state_old.orientation_cam;
  const Vector3d& t_c_w_old = imu_state_old.position_cam;

  // Feature position in old camera frame.
  Vector3d p_old;
  if (if_FEJ) {
    p_old = R_b2c*(R_b2w_old.transpose()*(feature.position_FEJ-imu_state_old.position_FEJ)-t_c_b);
  } else
    p_old = R_c2w_old.inverse() * (p_w-t_c_w_old);

  // Poses of new camera and corresponding imu.
  const IMUState_Aug& imu_state_new = state_server.imu_states_augment[old_state_id];
  Matrix3d R_b2w_new = imu_state_new.orientation;

  Matrix3d R_w2b_new = R_b2w_new.transpose();
  const Vector3d& t_b_w_new = imu_state_new.position;
  Matrix3d R_c2w_new = imu_state_new.orientation_cam;
  Matrix3d R_w2c_new = R_c2w_new.transpose();

  // Feature position in new camera frame.
  const Vector3d& inv_new = feature.invParam;

  // A commonly used vector
  Vector3d p_bf_w_old, p_bf_w_new;
  if (if_FEJ) {
    p_bf_w_old = feature.position_FEJ-imu_state_old.position_FEJ;
    p_bf_w_new = feature.position_FEJ-imu_state_new.position_FEJ;
  } else {
    p_bf_w_old = p_w-t_b_w_old;
    p_bf_w_new = p_w-t_b_w_new;
  }

  // Calculate Jacobians.
  Matrix3d J_fp_new = Matrix3d::Identity();
  J_fp_new(0, 2) = -inv_new(0);
  J_fp_new(1, 2) = -inv_new(1);
  J_fp_new(2, 2) = -inv_new(2);
  J_fp_new = inv_new(2)*J_fp_new;

  Matrix3d J_p = R_w2c_new * R_c2w_old;
  MatrixXd J_x_old = MatrixXd::Zero(3, 6);
  J_x_old.leftCols(3) = -R_w2c_new * skewSymmetric(p_bf_w_old);
  J_x_old.rightCols(3) = R_w2c_new;
  MatrixXd J_x_new = MatrixXd::Zero(3, 6);
  J_x_new.leftCols(3) = R_w2c_new * skewSymmetric(p_bf_w_new);
  J_x_new.rightCols(3) = -R_w2c_new;
  MatrixXd J_e = MatrixXd::Zero(3, 6);
  Matrix3d SkewMx = skewSymmetric(R_w2b_new*p_bf_w_new-t_c_b);
  Matrix3d Mx = R_w2b_new * R_b2w_old * skewSymmetric(R_b2c.transpose()*p_old);
  J_e.leftCols(3) = R_b2c * (SkewMx-Mx);
  J_e.rightCols(3) = R_b2c * (R_w2b_new*R_b2w_old-Matrix3d::Identity());

  Matrix3d J_pf_old = Matrix3d::Identity();
  J_pf_old(0, 2) = -p_old(0);
  J_pf_old(1, 2) = -p_old(1);
  J_pf_old(2, 2) = -p_old(2);
  J_pf_old = p_old(2)*J_pf_old;

  Matrix3d H_f_new = J_fp_new*J_p*J_pf_old;
  MatrixXd H_x_old = J_fp_new*J_x_old;
  MatrixXd H_x_new = J_fp_new*J_x_new;
  MatrixXd H_e = J_fp_new*J_e;

  // Stack Jacobians.
  MatrixXd J = MatrixXd::Zero(3, state_server.state_cov.cols());
  auto old_state_iter = state_server.imu_states_augment.find(old_state_id);
  int old_state_cntr = std::distance(
          state_server.imu_states_augment.begin(), old_state_iter);
  auto new_state_iter = state_server.imu_states_augment.find(old_state_id);
  int new_state_cntr = std::distance(
          state_server.imu_states_augment.begin(), new_state_iter);
  auto feature_iter = find(state_server.feature_states.begin(),
          state_server.feature_states.end(), feature_id);
  int feature_cntr = std::distance(
          state_server.feature_states.begin(), feature_iter);
  J.block<3, 3>(0, LEG_DIM+6*state_size+3*feature_cntr) = H_f_new;
  J.block<3, 6>(0, LEG_DIM+6*old_state_cntr) = H_x_old;
  J.block<3, 6>(0, LEG_DIM+6*new_state_cntr) = H_x_new;
  J.block<3, 6>(0, 15) = H_e;

  // Compute new covariance portions.
  MatrixXd Pfleg = J * state_server.state_cov;
  Matrix3d Pff = Pfleg * J.transpose();
  MatrixXd Pfleg_left = Pfleg.block(
    0, 0, 3, LEG_DIM+6*state_size+3*feature_cntr);
  MatrixXd Pfleg_right;
  if (use_schmidt) {
    Pfleg_right = Pfleg.block(
      0, LEG_DIM+6*state_size+3*feature_cntr+3,
      3, 3*feature_size-3*feature_cntr-3+6*state_server.nui_ids.size());
  } else {
    Pfleg_right = Pfleg.block(
      0, LEG_DIM+6*state_size+3*feature_cntr+3,
      3, 3*feature_size-3*feature_cntr-3);
  }

  // Update covariance.
  state_server.state_cov.block(
    LEG_DIM+6*state_size+3*feature_cntr,
    LEG_DIM+6*state_size+3*feature_cntr,
    3, 3) = Pff;
  state_server.state_cov.block(
    LEG_DIM+6*state_size+3*feature_cntr, 0,
    3, LEG_DIM+6*state_size+3*feature_cntr) = Pfleg_left;
  state_server.state_cov.block(
    0, LEG_DIM+6*state_size+3*feature_cntr,
    LEG_DIM+6*state_size+3*feature_cntr, 3) = Pfleg_left.transpose();
  if (use_schmidt) {
    state_server.state_cov.block(
      LEG_DIM+6*state_size+3*feature_cntr,
      LEG_DIM+6*state_size+3*feature_cntr+3,
      3, 3*feature_size-3*feature_cntr-3+6*state_server.nui_ids.size()) = Pfleg_right;
    state_server.state_cov.block(
      LEG_DIM+6*state_size+3*feature_cntr+3,
      LEG_DIM+6*state_size+3*feature_cntr,
      3*feature_size-3*feature_cntr-3+6*state_server.nui_ids.size(), 3) = Pfleg_right.transpose();
  } else {
    state_server.state_cov.block(
      LEG_DIM+6*state_size+3*feature_cntr,
      LEG_DIM+6*state_size+3*feature_cntr+3,
      3, 3*feature_size-3*feature_cntr-3) = Pfleg_right;
    state_server.state_cov.block(
      LEG_DIM+6*state_size+3*feature_cntr+3,
      LEG_DIM+6*state_size+3*feature_cntr,
      3*feature_size-3*feature_cntr-3, 3) = Pfleg_right.transpose();
  }

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}


void OrcVIO::updateFeatureCov_1didp(const FeatureIDType& feature_id,
        const StateIDType& old_state_id, const StateIDType& new_state_id) {
  const size_t state_size = state_server.imu_states_augment.size();
  const size_t feature_size = state_server.feature_states.size();

  const Feature& feature = map_server[feature_id];

  // Feature position in world frame.
  const Vector3d& p_w = feature.position;

  // Imu-camera extrinsic.
  const Matrix3d& R_b2c = state_server.imu_state.R_imu_cam0;
  const Vector3d& t_c_b = state_server.imu_state.t_cam0_imu;

  // Poses of old camera and corresponding imu.
  const IMUState_Aug& imu_state_old = state_server.imu_states_augment[old_state_id];
  Matrix3d R_b2w_old = imu_state_old.orientation;

  const Vector3d& t_b_w_old = imu_state_old.position;
  Matrix3d R_c2w_old = imu_state_old.orientation_cam;
  const Vector3d& t_c_w_old = imu_state_old.position_cam;

  // Feature position in old camera frame.
  Vector3d p_old;
  if (if_FEJ) {
    p_old = R_b2c*(R_b2w_old.transpose()*(feature.position_FEJ-imu_state_old.position_FEJ)-t_c_b);
  } else
    p_old = R_c2w_old.inverse() * (p_w-t_c_w_old);

  // Inverse depth and Corrected observation in old camera frame.
  Vector3d p_old_ = R_c2w_old.inverse() * (p_w-t_c_w_old);
  double invDepth_old = 1/p_old_(2);
  Vector3d f_old = Vector3d(p_old_(0)/p_old_(2),
      p_old_(1)/p_old_(2), 1);

  // Poses of new camera and corresponding imu.
  const IMUState_Aug& imu_state_new = state_server.imu_states_augment[new_state_id];
  Matrix3d R_b2w_new = imu_state_new.orientation;

  Matrix3d R_w2b_new = R_b2w_new.transpose();
  const Vector3d& t_b_w_new = imu_state_new.position;
  Matrix3d R_c2w_new = imu_state_new.orientation_cam;
  Matrix3d R_w2c_new = R_c2w_new.transpose();

  // Inverse depth in new camera frame.
  const double& invDepth_new = feature.invDepth;

  // A commonly used vector
  Vector3d p_bf_w_old, p_bf_w_new;
  if (if_FEJ) {
    p_bf_w_old = feature.position_FEJ-imu_state_old.position_FEJ;
    p_bf_w_new = feature.position_FEJ-imu_state_new.position_FEJ;
  } else {
    p_bf_w_old = p_w-t_b_w_old;
    p_bf_w_new = p_w-t_b_w_new;
  }

  // Calculate Jacobians.
  double J_rho_d_new = -invDepth_new*invDepth_new;

  Vector3d J_d_ = R_w2c_new * R_c2w_old * f_old;
  double J_d = J_d_(2);

  Matrix3d J_theta_old_ = -R_w2c_new * skewSymmetric(p_bf_w_old);
  Matrix<double,1,3> J_theta_old = J_theta_old_.bottomRows(1);
  Matrix3d J_p_old_ = R_w2c_new;
  Matrix<double,1,3> J_p_old = J_p_old_.bottomRows(1);

  Matrix3d J_theta_new_ = R_w2c_new * skewSymmetric(p_bf_w_new);
  Matrix<double,1,3> J_theta_new = J_theta_new_.bottomRows(1);
  Matrix3d J_p_new_ = -R_w2c_new;
  Matrix<double,1,3> J_p_new = J_p_new_.bottomRows(1);

  Matrix3d SkewMx = skewSymmetric(R_w2b_new*p_bf_w_new-t_c_b);
  Matrix3d Mx = R_w2b_new * R_b2w_old * skewSymmetric(R_b2c.transpose()*p_old);
  Matrix3d J_e_theta_ = R_b2c * (SkewMx-Mx);
  Matrix<double,1,3> J_e_theta = J_e_theta_.bottomRows(1);
  Matrix3d J_e_p_ = R_b2c * (R_w2b_new*R_b2w_old-Matrix3d::Identity());
  Matrix<double,1,3> J_e_p = J_e_p_.bottomRows(1);

  double J_d_rho_old = -1/(invDepth_old*invDepth_old);

  double H_f_new = J_rho_d_new*J_d*J_d_rho_old;
  Matrix<double,1,3> H_theta_old = J_rho_d_new*J_theta_old;
  Matrix<double,1,3> H_p_old = J_rho_d_new*J_p_old;
  Matrix<double,1,3> H_theta_new = J_rho_d_new*J_theta_new;
  Matrix<double,1,3> H_p_new = J_rho_d_new*J_p_new;
  Matrix<double,1,3> H_e_theta = J_rho_d_new*J_e_theta;
  Matrix<double,1,3> H_e_p = J_rho_d_new*J_e_p;

  // Stack Jacobians.
  MatrixXd J = MatrixXd::Zero(1, state_server.state_cov.cols());
  auto old_state_iter = state_server.imu_states_augment.find(old_state_id);
  int old_state_cntr = std::distance(
          state_server.imu_states_augment.begin(), old_state_iter);
  auto new_state_iter = state_server.imu_states_augment.find(new_state_id);
  int new_state_cntr = std::distance(
          state_server.imu_states_augment.begin(), new_state_iter);
  auto feature_iter = find(state_server.feature_states.begin(),
          state_server.feature_states.end(), feature_id);
  int feature_cntr = std::distance(
          state_server.feature_states.begin(), feature_iter);
  J(0, LEG_DIM+6*state_size+feature_cntr) = H_f_new;
  J.block<1, 3>(0, LEG_DIM+6*old_state_cntr) = H_theta_old;
  J.block<1, 3>(0, LEG_DIM+6*old_state_cntr+3) = H_p_old;
  J.block<1, 3>(0, LEG_DIM+6*new_state_cntr) = H_theta_new;
  J.block<1, 3>(0, LEG_DIM+6*new_state_cntr+3) = H_p_new;
  J.block<1, 3>(0, 15) = H_e_theta;
  J.block<1, 3>(0, 18) = H_e_p;

  // Compute new covariance portions.
  MatrixXd Pfleg = J * state_server.state_cov;
  MatrixXd Pff = Pfleg * J.transpose();
  MatrixXd Pfleg_left = Pfleg.block(
    0, 0, 1, LEG_DIM+6*state_size+feature_cntr);
  MatrixXd Pfleg_right;
  if (use_schmidt) {
    Pfleg_right = Pfleg.block(
      0, LEG_DIM+6*state_size+feature_cntr+1,
      1, feature_size-feature_cntr-1+6*state_server.nui_ids.size());
  } else {
    Pfleg_right = Pfleg.block(
      0, LEG_DIM+6*state_size+feature_cntr+1,
      1, feature_size-feature_cntr-1);
  }

  // Update covariance.
  state_server.state_cov.block(
    LEG_DIM+6*state_size+feature_cntr,
    LEG_DIM+6*state_size+feature_cntr,
    1, 1) = Pff;
  state_server.state_cov.block(
    LEG_DIM+6*state_size+feature_cntr, 0,
    1, LEG_DIM+6*state_size+feature_cntr) = Pfleg_left;
  state_server.state_cov.block(
    0, LEG_DIM+6*state_size+feature_cntr,
    LEG_DIM+6*state_size+feature_cntr, 1) = Pfleg_left.transpose();
  if (use_schmidt) {
    state_server.state_cov.block(
      LEG_DIM+6*state_size+feature_cntr,
      LEG_DIM+6*state_size+feature_cntr+1,
      1, feature_size-feature_cntr-1+6*state_server.nui_ids.size()) = Pfleg_right;
    state_server.state_cov.block(
      LEG_DIM+6*state_size+feature_cntr+1,
      LEG_DIM+6*state_size+feature_cntr,
      feature_size-feature_cntr-1+6*state_server.nui_ids.size(), 1) = Pfleg_right.transpose();
  } else {
    state_server.state_cov.block(
      LEG_DIM+6*state_size+feature_cntr,
      LEG_DIM+6*state_size+feature_cntr+1,
      1, feature_size-feature_cntr-1) = Pfleg_right;
    state_server.state_cov.block(
      LEG_DIM+6*state_size+feature_cntr+1,
      LEG_DIM+6*state_size+feature_cntr,
      feature_size-feature_cntr-1, 1) = Pfleg_right.transpose();
  }

  // Fix the covariance to be symmetric
  MatrixXd state_cov_fixed = (state_server.state_cov + state_server.state_cov.transpose()) / 2.0;
  state_server.state_cov = state_cov_fixed;

  return;
}


void OrcVIO::rmLostFeaturesCov(const std::vector<FeatureIDType>& lost_ids) {
  if (lost_ids.empty())
    return;

  for (const auto& feature_id : lost_ids) {
    // Delete this feature, remove its covariance matrix;
    auto feature_itr = find(state_server.feature_states.begin(),
          state_server.feature_states.end(), feature_id);
    int feature_sequence = std::distance(
          state_server.feature_states.begin(), feature_itr);
    int feature_state_start = LEG_DIM + 6*state_server.imu_states_augment.size() + feature_idp_dim*feature_sequence;
    int feature_state_end = feature_state_start + feature_idp_dim;

    // Remove the corresponding rows and columns in the state
    // covariance matrix.
    if (feature_state_end < state_server.state_cov.rows()) {
      state_server.state_cov.block(feature_state_start, 0,
          state_server.state_cov.rows()-feature_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(feature_state_end, 0,
            state_server.state_cov.rows()-feature_state_end,
            state_server.state_cov.cols());

      state_server.state_cov.block(0, feature_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-feature_state_end) =
        state_server.state_cov.block(0, feature_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-feature_state_end);
    }
    state_server.state_cov.conservativeResize(
        state_server.state_cov.rows()-feature_idp_dim, state_server.state_cov.cols()-feature_idp_dim);

    // Remove this feature from its nuisance anchor if needed
    if (use_schmidt) {
      StateIDType id_an = map_server[feature_id].id_anchor;
      if (find(state_server.nui_ids.begin(),
            state_server.nui_ids.end(), id_an)
          != state_server.nui_ids.end()) {
        auto itr = find(state_server.nui_features[id_an].begin(),
                  state_server.nui_features[id_an].end(), feature_id);
        state_server.nui_features[id_an].erase(itr);
      }
    }

    // publish lost ekf features
    lost_slam_features[feature_id] = map_server[feature_id];

    // Remove this feature state from the state vector and the map
    state_server.feature_states.erase(feature_itr);
    map_server.erase(feature_id);
  }
}


void OrcVIO::updateGridMap() {
  if (0==grid_rows*grid_cols)
    return;

  // Reset grid map
  for (int i=0; i<grid_rows*grid_cols; ++i)
    grid_map[i] = vector<FeatureIDType>(0);

  // Update grid map
  for (auto feature_id : state_server.feature_states) {
    Vector2d xy =
      map_server[feature_id].observations[state_server.imu_state.id];
    int row = static_cast<int>((xy(1)-y_min)/grid_height);
    int col = static_cast<int>((xy(0)-x_min)/grid_width);
    int code = row*grid_cols + col;
    grid_map[code].push_back(feature_id);
  }

  return;
}


void OrcVIO::delRedundantFeatures() {
  if (0==grid_rows*grid_cols*max_features)
    return;

  // Filter out reduntant features
  vector<FeatureIDType> rm_feature_ids(0);
  for (int i=0; i<grid_rows*grid_cols; ++i) {
    const auto& v_IDs = grid_map[i];
    if (v_IDs.size()<=max_features)
      continue;
    vector<pair<FeatureIDType,int>> id_num(0);
    for (int j=0; j<v_IDs.size(); ++j) {
      const FeatureIDType& id = v_IDs[j];
      id_num.push_back(
        make_pair(id, map_server[id].totalObsNum));
      // debug log
      auto feature_iter = find(state_server.feature_states.begin(),
        state_server.feature_states.end(), id);
      if (feature_iter == state_server.feature_states.end())
        printf("AN UNEXPECTED ERROR HAPPENED !");
    }
    sort(id_num.begin(), id_num.end(),
          OrcVIO::compareByObsNum);
    auto it = id_num.begin();
    for (int j=0; j<v_IDs.size()-max_features;
        ++j, ++it)
      rm_feature_ids.push_back(it->first);
  }

  // Delete redundant ekf features
  rmLostFeaturesCov(rm_feature_ids);

  // Update information in grid map
  // updateGridMap();

  return;
}


StateIDType OrcVIO::getNewAnchorId(Feature& feature, const std::vector<StateIDType>& rmIDs) {
  FeatureIDType id = feature.id;
  Vector3d p_w = feature.position;

  auto key_state_iter = state_server.imu_states_augment.end();
  int halfSize = (int)(state_server.imu_states_augment.size()/2);
  int size = state_server.imu_states_augment.size();
  for (int i = 0; i < size; ++i)
    --key_state_iter;
  StateIDType keyID = key_state_iter->first;

  int looplen;
  if (size<=2) {
    printf("Size of imu_states_augment is not big enough!");
    key_state_iter = state_server.imu_states_augment.end();
    --key_state_iter;
    return key_state_iter->first;
  } else {
    looplen = size-2;
  }

  bool bValid = false;
  double minDis = 99999;
  double Id_min;
  for (int i = 0; i < looplen; ++i, ++key_state_iter) {
    bool bObs = (feature.observations.find(key_state_iter->first)
                != feature.observations.end());
    if (!bObs)
      continue;
    auto itr = find(rmIDs.begin(), rmIDs.end(), key_state_iter->first);
    bool bNotRm = (itr == rmIDs.end());
    if (bNotRm) {
      Matrix3d R_c2w = key_state_iter->second.orientation_cam;
      const Vector3d& t_c_w = key_state_iter->second.position_cam;
      Vector3d p_new = R_c2w.inverse() * (p_w-t_c_w);
      Vector2d vdis(p_new(0)/p_new(2)-feature.observations[key_state_iter->first](0),
                    p_new(1)/p_new(2)-feature.observations[key_state_iter->first](1));
      double dis = vdis.norm();
      if (minDis>dis) {
        minDis = dis;
        Id_min = key_state_iter->first;
        bValid = true;
      }
    }
  }

  // debug log
  if (key_state_iter==state_server.imu_states_augment.end())
    printf("new anchor id should not added to the end!");

  if (bValid) {
    return Id_min;
  }
  else {
    key_state_iter = state_server.imu_states_augment.end();
    --key_state_iter;
    return key_state_iter->first;
  }
}

void OrcVIO::calPhiEulerMethod(Eigen::MatrixXd& Phi, const double& dtime,
        const Eigen::Vector3d& acc_hat, const Eigen::Vector3d& gyro_hat) {

    Eigen::Matrix3d wRi = state_server.imu_state.orientation;

    Phi = MatrixXd::Identity(LEG_DIM, LEG_DIM);

    // note that the orcvio has different order of position and velocity 
    if (use_left_perturbation_flag)
    {
      Phi.block<3, 3>(0, 9) = -dtime * wRi;
      Phi.block<3, 3>(3, 0) = -dtime * skewSymmetric(wRi * acc_hat);
      Phi.block<3, 3>(3, 12) = -dtime * wRi;
      Phi.block<3, 3>(6, 3) = dtime * Matrix3d::Identity();
    }
    else 
    {
      Phi.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() - dtime * skewSymmetric(gyro_hat);
      Phi.block<3, 3>(0, 9) = -dtime * Eigen::Matrix3d::Identity();
      Phi.block<3, 3>(3, 0) = -dtime * wRi * skewSymmetric(acc_hat);
      Phi.block<3, 3>(3, 12) = -dtime * wRi;
      Phi.block<3, 3>(6, 3) = dtime * Eigen::Matrix3d::Identity();
    }

    return;

}

void OrcVIO::calPhiClosedForm(Eigen::MatrixXd& Phi, const double& dtime,
        const Eigen::Vector3d& f, const Eigen::Vector3d& w, 
        const Eigen::Vector3d& acc, const Eigen::Vector3d& gyro, 
        const Eigen::Vector3d& f_old, const Eigen::Vector3d& w_old, 
        const Eigen::Vector3d& acc_old, const Eigen::Vector3d& gyro_old) {

  Phi = MatrixXd::Identity(LEG_DIM, LEG_DIM);

  Vector3d f_mid = (f+f_old)/2;
  Vector3d acc_mid = (acc+acc_old)/2;
  Vector3d w_mid = (w_old+w)/2 + dtime*(w_old.cross(w))/12;

  Vector3d Axis_Angle = dtime*(gyro_old+gyro)/2 + dtime*dtime*(gyro_old.cross(gyro))/12;
  Matrix3d AxisAngle_hat = skewSymmetric(Axis_Angle);

  Matrix3d C_bk2w = state_server.imu_state_old.orientation;

  if (use_larvio_flag || use_left_perturbation_flag)
  {

    Matrix3d TA = state_server.Tg*state_server.As;

    // Simple blocks
    Vector3d vk = (if_FEJ ?
        state_server.imu_state_FEJ_old.velocity :
        state_server.imu_state_old.velocity);
    Vector3d pk = (if_FEJ ?
        state_server.imu_state_FEJ_old.position :
        state_server.imu_state_old.position);
    Vector3d vkp1 = (if_FEJ ?
        state_server.imu_state_FEJ_now.velocity :
        state_server.imu_state.velocity);
    Vector3d pkp1 = (if_FEJ ?
        state_server.imu_state_FEJ_now.position :
        state_server.imu_state.position);
    Vector3d g_w = IMUState::gravity;
    Phi.block<3, 3>(0, 9) =     // Phi_q_bg
            -0.5*C_bk2w*(2*Matrix3d::Identity()+AxisAngle_hat)*dtime*state_server.Tg;
    Phi.block<3, 3>(0, 12) =    // Phi_q_ba
            0.5*C_bk2w*(2*Matrix3d::Identity()+AxisAngle_hat)*dtime*TA*state_server.Ma;
    Phi.block<3, 3>(3, 0) =     // Phi_v_q
            -skewSymmetric(vkp1-vk-g_w*dtime);
    Phi.block<3, 3>(3, 9) =     // Phi_v_bg
            skewSymmetric(-pkp1+pk+vkp1*dtime-0.5*g_w*dtime*dtime)*C_bk2w +
            skewSymmetric(-0.5*pkp1+0.5*pk+0.5*vkp1*dtime-g_w*dtime*dtime/6)*C_bk2w*AxisAngle_hat;
    Phi.block<3, 3>(3, 12) =    // Phi_v_ba
            -0.5*C_bk2w*(2*Matrix3d::Identity()+AxisAngle_hat)*dtime*state_server.Ma -
            Phi.block<3,3>(3,9)*TA*state_server.Ma;
    Phi.block<3, 3>(6, 0) =     // Phi_p_q
            -skewSymmetric(pkp1-pk-vk*dtime-0.5*g_w*dtime*dtime);
    Phi.block<3, 3>(6, 3) =     // Phi_p_v
            Matrix3d::Identity()*dtime;
    Phi.block<3, 3>(6, 9) =     // Phi_p_bg
            -dtime*dtime*dtime*skewSymmetric(g_w)*C_bk2w/6 +
            dtime*skewSymmetric(pkp1-pk-g_w*dtime*dtime/6)*C_bk2w*AxisAngle_hat/4;
    Phi.block<3, 3>(6, 12) =    // Phi_p_ba
            -C_bk2w*(3*Matrix3d::Identity()+AxisAngle_hat)*dtime*dtime/6*state_server.Ma -
            Phi.block<3,3>(6,9)*TA*state_server.Ma;

    // Fill in corresponding blocks if calibrate IMU instrinsic
    if (calib_imu) {
      // Symbol simplification
      Matrix3d WL_k = Matrix3d::Zero();
      WL_k(1,0) = w_old(0);
      WL_k(2,1) = w_old(0);
      WL_k(2,2) = w_old(1);
      Matrix3d WD_k = Matrix3d::Zero();
      WD_k(0,0) = w_old(0);
      WD_k(1,1) = w_old(1);
      WD_k(2,2) = w_old(2);
      Matrix3d WU_k = Matrix3d::Zero();
      WU_k(0,0) = w_old(1);
      WU_k(0,1) = w_old(2);
      WU_k(1,2) = w_old(2);
      Matrix3d ML_k = Matrix3d::Zero();
      ML_k(1,0) = acc_old(0);
      ML_k(2,1) = acc_old(0);
      ML_k(2,2) = acc_old(1);
      Matrix3d MD_k = Matrix3d::Zero();
      MD_k(0,0) = acc_old(0);
      MD_k(1,1) = acc_old(1);
      MD_k(2,2) = acc_old(2);
      Matrix3d MU_k = Matrix3d::Zero();
      MU_k(0,0) = acc_old(1);
      MU_k(0,1) = acc_old(2);
      MU_k(1,2) = acc_old(2);
      Matrix3d FL_k = Matrix3d::Zero();
      FL_k(1,0) = f_old(0);
      FL_k(2,1) = f_old(0);
      FL_k(2,2) = f_old(1);
      Matrix3d FD_k = Matrix3d::Zero();
      FD_k(0,0) = f_old(0);
      FD_k(1,1) = f_old(1);
      FD_k(2,2) = f_old(2);
      Matrix3d WL_kh = Matrix3d::Zero();
      WL_kh(1,0) = w_mid(0);
      WL_kh(2,1) = w_mid(0);
      WL_kh(2,2) = w_mid(1);
      Matrix3d WD_kh = Matrix3d::Zero();
      WD_kh(0,0) = w_mid(0);
      WD_kh(1,1) = w_mid(1);
      WD_kh(2,2) = w_mid(2);
      Matrix3d WU_kh = Matrix3d::Zero();
      WU_kh(0,0) = w_mid(1);
      WU_kh(0,1) = w_mid(2);
      WU_kh(1,2) = w_mid(2);
      Matrix3d ML_kh = Matrix3d::Zero();
      ML_kh(1,0) = acc_mid(0);
      ML_kh(2,1) = acc_mid(0);
      ML_kh(2,2) = acc_mid(1);
      Matrix3d MD_kh = Matrix3d::Zero();
      MD_kh(0,0) = acc_mid(0);
      MD_kh(1,1) = acc_mid(1);
      MD_kh(2,2) = acc_mid(2);
      Matrix3d MU_kh = Matrix3d::Zero();
      MU_kh(0,0) = acc_mid(1);
      MU_kh(0,1) = acc_mid(2);
      MU_kh(1,2) = acc_mid(2);
      Matrix3d FL_kh = Matrix3d::Zero();
      FL_kh(1,0) = f_mid(0);
      FL_kh(2,1) = f_mid(0);
      FL_kh(2,2) = f_mid(1);
      Matrix3d FD_kh = Matrix3d::Zero();
      FD_kh(0,0) = f_mid(0);
      FD_kh(1,1) = f_mid(1);
      FD_kh(2,2) = f_mid(2);
      Matrix3d WL_kp1 = Matrix3d::Zero();
      WL_kp1(1,0) = w(0);
      WL_kp1(2,1) = w(0);
      WL_kp1(2,2) = w(1);
      Matrix3d WD_kp1 = Matrix3d::Zero();
      WD_kp1(0,0) = w(0);
      WD_kp1(1,1) = w(1);
      WD_kp1(2,2) = w(2);
      Matrix3d WU_kp1 = Matrix3d::Zero();
      WU_kp1(0,0) = w(1);
      WU_kp1(0,1) = w(2);
      WU_kp1(1,2) = w(2);
      Matrix3d ML_kp1 = Matrix3d::Zero();
      ML_kp1(1,0) = acc(0);
      ML_kp1(2,1) = acc(0);
      ML_kp1(2,2) = acc(1);
      Matrix3d MD_kp1 = Matrix3d::Zero();
      MD_kp1(0,0) = acc(0);
      MD_kp1(1,1) = acc(1);
      MD_kp1(2,2) = acc(2);
      Matrix3d MU_kp1 = Matrix3d::Zero();
      MU_kp1(0,0) = acc(1);
      MU_kp1(0,1) = acc(2);
      MU_kp1(1,2) = acc(2);
      Matrix3d FL_kp1 = Matrix3d::Zero();
      FL_kp1(1,0) = f(0);
      FL_kp1(2,1) = f(0);
      FL_kp1(2,2) = f(1);
      Matrix3d FD_kp1 = Matrix3d::Zero();
      FD_kp1(0,0) = f(0);
      FD_kp1(1,1) = f(1);
      FD_kp1(2,2) = f(2);
      Matrix3d I = Matrix3d::Identity();
      Matrix3d R_mid = (I+0.5*AxisAngle_hat);
      Matrix3d R_kp1 = (I+AxisAngle_hat);

      // Tricky blocks about q
      Matrix3d kqT1_1 = WL_k;
      Matrix3d kqT1_2 = R_mid*WL_kh;
      Matrix3d kqT1_4 = R_kp1*WL_kp1;
      Matrix3d RWL_ = dtime*(kqT1_1+4*kqT1_2+kqT1_4)/6;
      Phi.block<3, 3>(0, 22) =    // Phi_q_T1
              C_bk2w*RWL_;
      Matrix3d kqT2_1 = WD_k;
      Matrix3d kqT2_2 = R_mid*WD_kh;
      Matrix3d kqT2_4 = R_kp1*WD_kp1;
      Matrix3d RWD_ = dtime*(kqT2_1+4*kqT2_2+kqT2_4)/6;
      Phi.block<3, 3>(0, 25) =    // Phi_q_T2
              C_bk2w*RWD_;
      Matrix3d kqT3_1 = WU_k;
      Matrix3d kqT3_2 = R_mid*WU_kh;
      Matrix3d kqT3_4 = R_kp1*WU_kp1;
      Matrix3d RWU_ = dtime*(kqT3_1+4*kqT3_2+kqT3_4)/6;
      Phi.block<3, 3>(0, 28) =    // Phi_q_T3
              C_bk2w*RWU_;
      Matrix3d kqA1_1 = state_server.Tg*ML_k;
      Matrix3d kqA1_2 = R_mid*state_server.Tg*ML_kh;
      Matrix3d kqA1_4 = R_kp1*state_server.Tg*ML_kp1;
      Matrix3d RML_ = dtime*(kqA1_1+4*kqA1_2+kqA1_4)/6;
      Phi.block<3, 3>(0, 31) =    // Phi_q_A1
              -C_bk2w*RML_;
      Matrix3d kqA2_1 = state_server.Tg*MD_k;
      Matrix3d kqA2_2 = R_mid*state_server.Tg*MD_kh;
      Matrix3d kqA2_4 = R_kp1*state_server.Tg*MD_kp1;
      Matrix3d RMD_ = dtime*(kqA2_1+4*kqA2_2+kqA2_4)/6;
      Phi.block<3, 3>(0, 34) =    // Phi_q_A2
              -C_bk2w*RMD_;
      Matrix3d kqA3_1 = state_server.Tg*MU_k;
      Matrix3d kqA3_2 = R_mid*state_server.Tg*MU_kh;
      Matrix3d kqA3_4 = R_kp1*state_server.Tg*MU_kp1;
      Matrix3d RMU_ = dtime*(kqA3_1+4*kqA3_2+kqA3_4)/6;
      Phi.block<3, 3>(0, 37) =    // Phi_q_A3
              -C_bk2w*RMU_;
      Matrix3d kqM1_1 = TA*FL_k;
      Matrix3d kqM1_2 = R_mid*TA*FL_kh;
      Matrix3d kqM1_4 = R_kp1*TA*FL_kp1;
      Matrix3d RFL_ = dtime*(kqM1_1+4*kqM1_2+kqM1_4)/6;
      Phi.block<3, 3>(0, 40) =    // Phi_q_M1
              -C_bk2w*RFL_;
      Matrix3d kqM2_1 = TA*FD_k;
      Matrix3d kqM2_2 = R_mid*TA*FD_kh;
      Matrix3d kqM2_4 = R_kp1*TA*FD_kp1;
      Matrix3d RFD_ = dtime*(kqM2_1+4*kqM2_2+kqM2_4)/6;
      Phi.block<3, 3>(0, 43) =    // Phi_q_M2
              -C_bk2w*RFD_;

      // Tricky blocks about v
      Matrix3d kvT1_1 = Matrix3d::Zero();
      Matrix3d kvT1_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqT1_1/2;
      Matrix3d kvT1_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqT1_2/2;
      Matrix3d kvT1_4 = skewSymmetric(R_kp1*acc)*RWL_;
      Matrix3d fRWL_ = dtime*(kvT1_1+2*kvT1_2+2*kvT1_3+kvT1_4)/6;
      Phi.block<3, 3>(3, 22) =    // Phi_v_T1
              -C_bk2w*fRWL_;
      Matrix3d kvT2_1 = Matrix3d::Zero();
      Matrix3d kvT2_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqT2_1/2;
      Matrix3d kvT2_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqT2_2/2;
      Matrix3d kvT2_4 = skewSymmetric(R_kp1*acc)*RWD_;
      Matrix3d fRWD_ = dtime*(kvT2_1+2*kvT2_2+2*kvT2_3+kvT2_4)/6;
      Phi.block<3, 3>(3, 25) =    // Phi_v_T2
              -C_bk2w*fRWD_;
      Matrix3d kvT3_1 = Matrix3d::Zero();
      Matrix3d kvT3_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqT3_1/2;
      Matrix3d kvT3_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqT3_2/2;
      Matrix3d kvT3_4 = skewSymmetric(R_kp1*acc)*RWU_;
      Matrix3d fRWU_ = dtime*(kvT3_1+2*kvT3_2+2*kvT3_3+kvT3_4)/6;
      Phi.block<3, 3>(3, 28) =    // Phi_v_T3
              -C_bk2w*fRWU_;
      Matrix3d kvA1_1 = Matrix3d::Zero();
      Matrix3d kvA1_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqA1_1/2;
      Matrix3d kvA1_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqA1_2/2;
      Matrix3d kvA1_4 = skewSymmetric(R_kp1*acc)*RML_;
      Matrix3d fRML_ = dtime*(kvA1_1+2*kvA1_2+2*kvA1_3+kvA1_4)/6;
      Phi.block<3, 3>(3, 31) =    // Phi_v_A1
              C_bk2w*fRML_;
      Matrix3d kvA2_1 = Matrix3d::Zero();
      Matrix3d kvA2_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqA2_1/2;
      Matrix3d kvA2_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqA2_2/2;
      Matrix3d kvA2_4 = skewSymmetric(R_kp1*acc)*RMD_;
      Matrix3d fRMD_ = dtime*(kvA2_1+2*kvA2_2+2*kvA2_3+kvA2_4)/6;
      Phi.block<3, 3>(3, 34) =    // Phi_v_A2
              C_bk2w*fRMD_;
      Matrix3d kvA3_1 = Matrix3d::Zero();
      Matrix3d kvA3_2 = skewSymmetric(R_mid*acc_mid)*dtime*kqA3_1/2;
      Matrix3d kvA3_3 = skewSymmetric(R_mid*acc_mid)*dtime*kqA3_2/2;
      Matrix3d kvA3_4 = skewSymmetric(R_kp1*acc)*RMU_;
      Matrix3d fRMU_ = dtime*(kvA3_1+2*kvA3_2+2*kvA3_3+kvA3_4)/6;
      Phi.block<3, 3>(3, 37) =    // Phi_v_A3
              C_bk2w*fRMU_;
      Matrix3d kvM1_1 = FL_k;
      Matrix3d kvM1_2 = R_mid*FL_kh
              + skewSymmetric(R_mid*acc_mid)*dtime*kqM1_1/2;
      Matrix3d kvM1_3 = R_mid*FL_kh
              + skewSymmetric(R_mid*acc_mid)*dtime*kqM1_2/2;
      Matrix3d kvM1_4 = R_kp1*FL_kp1
              + skewSymmetric(R_kp1*acc)*RFL_;
      Matrix3d vM1_ = dtime*(kvM1_1+2*kvM1_2+2*kvM1_3+kvM1_4)/6;
      Phi.block<3, 3>(3, 40) =    // Phi_v_M1
              C_bk2w*vM1_;
      Matrix3d kvM2_1 = FD_k;
      Matrix3d kvM2_2 = R_mid*FD_kh
              + skewSymmetric(R_mid*acc_mid)*dtime*kqM2_1/2;
      Matrix3d kvM2_3 = R_mid*FD_kh
              + skewSymmetric(R_mid*acc_mid)*dtime*kqM2_2/2;
      Matrix3d kvM2_4 = R_kp1*FD_kp1
              + skewSymmetric(R_kp1*acc)*RFD_;
      Matrix3d vM2_ = dtime*(kvM2_1+2*kvM2_2+2*kvM2_3+kvM2_4)/6;
      Phi.block<3, 3>(3, 43) =    // Phi_v_M2
              C_bk2w*vM2_;

      // Tricky blocks about p
      Matrix3d kpT1_1 = Matrix3d::Zero();
      Matrix3d kpT1_2 = dtime*kvT1_1/2;
      Matrix3d kpT1_3 = dtime*kvT1_2/2;
      Matrix3d kpT1_4 = fRWL_;
      Phi.block<3, 3>(6, 22) =    // Phi_p_T1
              -C_bk2w*dtime*(kpT1_1+2*kpT1_2+2*kpT1_3+kpT1_4)/6;
      Matrix3d kpT2_1 = Matrix3d::Zero();
      Matrix3d kpT2_2 = dtime*kvT2_1/2;
      Matrix3d kpT2_3 = dtime*kvT2_2/2;
      Matrix3d kpT2_4 = fRWD_;
      Phi.block<3, 3>(6, 25) =    // Phi_p_T2
              -C_bk2w*dtime*(kpT2_1+2*kpT2_2+2*kpT2_3+kpT2_4)/6;
      Matrix3d kpT3_1 = Matrix3d::Zero();
      Matrix3d kpT3_2 = dtime*kvT3_1/2;
      Matrix3d kpT3_3 = dtime*kvT3_2/2;
      Matrix3d kpT3_4 = fRWU_;
      Phi.block<3, 3>(6, 28) =    // Phi_p_T3
              -C_bk2w*dtime*(kpT3_1+2*kpT3_2+2*kpT3_3+kpT3_4)/6;
      Matrix3d kpA1_1 = Matrix3d::Zero();
      Matrix3d kpA1_2 = dtime*kvA1_1/2;
      Matrix3d kpA1_3 = dtime*kvA1_2/2;
      Matrix3d kpA1_4 = fRML_;
      Phi.block<3, 3>(6, 31) =    // Phi_p_A1
              C_bk2w*dtime*(kpA1_1+2*kpA1_2+2*kpA1_3+kpA1_4)/6;
      Matrix3d kpA2_1 = Matrix3d::Zero();
      Matrix3d kpA2_2 = dtime*kvA2_1/2;
      Matrix3d kpA2_3 = dtime*kvA2_2/2;
      Matrix3d kpA2_4 = fRMD_;
      Phi.block<3, 3>(6, 34) =    // Phi_p_A2
              C_bk2w*dtime*(kpA2_1+2*kpA2_2+2*kpA2_3+kpA2_4)/6;
      Matrix3d kpA3_1 = Matrix3d::Zero();
      Matrix3d kpA3_2 = dtime*kvA3_1/2;
      Matrix3d kpA3_3 = dtime*kvA3_2/2;
      Matrix3d kpA3_4 = fRMU_;
      Phi.block<3, 3>(6, 37) =    // Phi_p_A3
              C_bk2w*dtime*(kpA3_1+2*kpA3_2+2*kpA3_3+kpA3_4)/6;
      Matrix3d kpM1_1 = Matrix3d::Zero();
      Matrix3d kpM1_2 = dtime*kvM1_1/2;
      Matrix3d kpM1_3 = dtime*kvM1_2/2;
      Matrix3d kpM1_4 = vM1_;
      Phi.block<3, 3>(6, 40) =    // Phi_p_M1
              C_bk2w*dtime*(kpM1_1+2*kpM1_2+2*kpM1_3+kpM1_4)/6;
      Matrix3d kpM2_1 = Matrix3d::Zero();
      Matrix3d kpM2_2 = dtime*kvM2_1/2;
      Matrix3d kpM2_3 = dtime*kvM2_2/2;
      Matrix3d kpM2_4 = vM2_;
      Phi.block<3, 3>(6, 43) =    // Phi_p_M2
              C_bk2w*dtime*(kpM2_1+2*kpM2_2+2*kpM2_3+kpM2_4)/6;
    }

  }
  else 
  {
    // we should always use acc in place (a_m - b_a) in Joan Sola's expressions and gyro in place of (w_m - b_m)
    // imu_state_old and imu_state are the same at this point
    // this function assumes that IMU intrinsics have trivial values

    // these two are equivalent
    // Matrix3d wRi = state_server.imu_state.orientation;
    Matrix3d wRi = C_bk2w;

    // Vector3d gyro_hat = Axis_Angle;
    Vector3d gyro_hat = gyro;

    // these two seem similar
    // Vector3d acc_hat = acc_mid;
    Vector3d acc_hat = acc;

    // prepare the basic terms 

    Matrix3d a_skew = skewSymmetric(acc_hat);
    Matrix3d g_skew = skewSymmetric(gyro_hat);
    double g_norm = gyro_hat.norm();

    Matrix3d theta_theta = Sophus::SO3d::exp(-dtime * gyro_hat).matrix();
    Matrix3d JL_plus = Jl_operator(dtime * gyro_hat);
    Matrix3d JL_minus = Jl_operator(-dtime * gyro_hat); 
    Matrix3d I3 = Matrix3d::Identity();
    Matrix3d Delta = -(g_skew / pow(g_norm, 2)) * (theta_theta.transpose() * (dtime * g_skew - I3) + I3);
    Matrix3d HL_plus = Hl_operator(dtime * gyro_hat);
    Matrix3d HL_minus = Hl_operator(-dtime * gyro_hat);

    // prepare the terms in Phi 
    Matrix3d theta_gyro = -dtime * JL_minus;

    Matrix3d v_theta = -dtime * wRi * skewSymmetric(JL_plus * acc_hat);
    Matrix3d v_gyro = wRi * Delta * a_skew * (I3 + (g_skew * g_skew / pow(g_norm, 2))) + dtime * wRi * JL_plus * (a_skew * g_skew / pow(g_norm, 2)) + dtime * wRi * (gyro_hat * acc_hat.transpose() / pow(g_norm, 2)) * JL_minus - dtime * ((acc_hat.transpose() * gyro_hat).value() / pow(g_norm, 2)) * I3;
    Matrix3d v_acc = -dtime * wRi * JL_plus;

    Matrix3d p_theta = -pow(dtime, 2) * wRi * skewSymmetric(HL_plus * acc_hat);
    Matrix3d p_v = dtime * I3; 
    Matrix3d p_gyro = wRi * (-g_skew * Delta - dtime * JL_plus + dtime * I3) * a_skew * (I3 + (g_skew * g_skew / pow(g_norm, 2))) * (g_skew / pow(g_norm, 2)) + pow(dtime, 2) * wRi * HL_plus * (a_skew * g_skew / pow(g_norm, 2)) + pow(dtime, 2) * wRi * (gyro_hat * acc_hat.transpose() / pow(g_norm, 2)) * HL_minus - pow(dtime, 2) * ((acc_hat.transpose() * gyro_hat).value() / (2*pow(g_norm, 2))) * wRi; 
    Matrix3d p_acc = -pow(dtime, 2) * wRi * HL_plus; 

    // for phi 
    
    // theta row 
    Phi.block<3, 3>(0, 0) = theta_theta;
    Phi.block<3, 3>(0, 9) = theta_gyro;

    // v row 
    Phi.block<3, 3>(3, 0) = v_theta;  
    Phi.block<3, 3>(3, 9) = v_gyro;
    Phi.block<3, 3>(3, 12) = v_acc;

    // p row 
    Phi.block<3, 3>(6, 0) = p_theta;
    Phi.block<3, 3>(6, 3) = p_v;
    Phi.block<3, 3>(6, 9) = p_gyro;
    Phi.block<3, 3>(6, 12) = p_acc;
  }

  return;
}


void OrcVIO::updateImuMx() {

  Matrix3d& Tg = state_server.Tg;
  Matrix3d& As = state_server.As;
  Matrix3d& Ma = state_server.Ma;
  Vector3d& T1 = state_server.T1;
  Vector3d& T2 = state_server.T2;
  Vector3d& T3 = state_server.T3;
  Vector3d& A1 = state_server.A1;
  Vector3d& A2 = state_server.A2;
  Vector3d& A3 = state_server.A3;
  Vector3d& M1 = state_server.M1;
  Vector3d& M2 = state_server.M2;

  // Tg
  Tg(0,0) = T2(0);
  Tg(0,1) = T3(0);
  Tg(0,2) = T3(1);
  Tg(1,0) = T1(0);
  Tg(1,1) = T2(1);
  Tg(1,2) = T3(2);
  Tg(2,0) = T1(1);
  Tg(2,1) = T1(2);
  Tg(2,2) = T2(2);

  // As
  As(0,0) = A2(0);
  As(0,1) = A3(0);
  As(0,2) = A3(1);
  As(1,0) = A1(0);
  As(1,1) = A2(1);
  As(1,2) = A3(2);
  As(2,0) = A1(1);
  As(2,1) = A1(2);
  As(2,2) = A2(2);

  // Ma
  Ma(0,0) = M2(0);
  Ma(1,0) = M1(0);
  Ma(1,1) = M2(1);
  Ma(2,0) = M1(1);
  Ma(2,1) = M1(2);
  Ma(2,2) = M2(2);

  return;
}


void OrcVIO::rmUselessNuisanceState() {
  // Pick out nuisance states to be deleted
  vector<StateIDType> rm_ids(0);
  for (int i=0; i<state_server.nui_ids.size(); ++i) {
    const StateIDType id_nui = state_server.nui_ids[i];
    if (0==state_server.nui_features[id_nui].size())
      rm_ids.push_back(id_nui);
  }
  // Delete relevant covariance blocks and information
  for (int i=0; i<rm_ids.size(); ++i) {
    const StateIDType id_nui = rm_ids[i];
    // Get index
    auto nui_itr = find(state_server.nui_ids.begin(),
          state_server.nui_ids.end(), id_nui);
    int nui_sequence = std::distance(
          state_server.nui_ids.begin(), nui_itr);
    int nui_state_start = LEG_DIM
        + 6*state_server.imu_states_augment.size()
        + feature_idp_dim*state_server.feature_states.size()
        + 6*nui_sequence;
    int nui_state_end = nui_state_start + 6;
    // Delete relevant matrix blocks
    if (nui_state_end<state_server.state_cov.cols()) {
      state_server.state_cov.block(nui_state_start, 0,
          state_server.state_cov.rows()-nui_state_end,
          state_server.state_cov.cols()) =
        state_server.state_cov.block(nui_state_end, 0,
            state_server.state_cov.rows()-nui_state_end,
            state_server.state_cov.cols());
      state_server.state_cov.block(0, nui_state_start,
          state_server.state_cov.rows(),
          state_server.state_cov.cols()-nui_state_end) =
        state_server.state_cov.block(0, nui_state_end,
            state_server.state_cov.rows(),
            state_server.state_cov.cols()-nui_state_end);
    }
    state_server.state_cov.conservativeResize(
      state_server.state_cov.rows()-6, state_server.state_cov.cols()-6);
    // Delete information in state_server
    state_server.nui_ids.erase(nui_itr);
    state_server.nui_imu_states.erase(id_nui);
    state_server.nui_features.erase(id_nui);
  }

  return;
}

void OrcVIO::incrementState_IMUCam(const VectorXd& delta_x)
{

  // Update the legacy state.
  // const VectorXd& delta_x_legacy = delta_x.head(LEG_DIM);
  VectorXd delta_x_legacy = delta_x.head(LEG_DIM);

  // TODO FIXME do not do this step for KITTI? 
  // Use this to handle fast rotation in KITTI 
  // printf("delta velocity: %f\n", delta_x_legacy.segment<3>(3).norm());
  // printf("delta position: %f\n", delta_x_legacy.segment<3>(6).norm());
  if ((//delta_x_legacy.segment<3>(0).norm() > 0.15 ||
      delta_x_legacy.segment<3>(3).norm() > 1.0 ||
      delta_x_legacy.segment<3>(6).norm() > 1.5 /*||
      delta_x_legacy.segment<3>(9).norm() > 0.15 ||
      delta_x_legacy.segment<3>(12).norm() > 0.5*/)
      && discard_large_update_flag
      ) {
    printf("delta velocity: %f\n", delta_x_legacy.segment<3>(3).norm());
    printf("delta position: %f\n", delta_x_legacy.segment<3>(6).norm());
    printf("in measurementUpdate_msckf: Update change is too large.\n");
    // do not do update on position and velocity 
    // delta_x_legacy.segment<3>(3) = Eigen::Vector3d::Zero();
    // delta_x_legacy.segment<3>(6) = Eigen::Vector3d::Zero();
    // do not do update at all 
    return; 
  }

  // Update IMU state
  Sophus::SO3d R_temp = Sophus::SO3d::exp(delta_x_legacy.head<3>());
  if (use_larvio_flag || use_left_perturbation_flag)
  {
    state_server.imu_state.orientation = R_temp.matrix() * state_server.imu_state.orientation;
  }
  else 
  {
    state_server.imu_state.orientation = state_server.imu_state.orientation * R_temp.matrix();
  }

  state_server.imu_state.velocity += delta_x_legacy.segment<3>(3);
  state_server.imu_state.position += delta_x_legacy.segment<3>(6);

  state_server.imu_state.gyro_bias += delta_x_legacy.segment<3>(9);
  state_server.imu_state.acc_bias += delta_x_legacy.segment<3>(12);

  // Update the IMU-CAM extrinsic
  const Vector4d dq_extrinsic = smallAngleQuaternion(delta_x_legacy.segment<3>(15));
  state_server.imu_state.R_imu_cam0 = state_server.imu_state.R_imu_cam0 *
      Quaterniond(dq_extrinsic(3),dq_extrinsic(0),dq_extrinsic(1),dq_extrinsic(2)).toRotationMatrix().transpose();
  state_server.imu_state.t_cam0_imu += delta_x_legacy.segment<3>(18);

  // Update td
  state_server.td += delta_x_legacy(21);

  // Update IMU instrinsic
  if (calib_imu) {
    state_server.T1 += delta_x_legacy.segment<3>(22);
    state_server.T2 += delta_x_legacy.segment<3>(25);
    state_server.T3 += delta_x_legacy.segment<3>(28);
    state_server.A1 += delta_x_legacy.segment<3>(31);
    state_server.A2 += delta_x_legacy.segment<3>(34);
    state_server.A3 += delta_x_legacy.segment<3>(37);
    state_server.M1 += delta_x_legacy.segment<3>(40);
    state_server.M2 += delta_x_legacy.segment<3>(43);
    updateImuMx();
  }

  // Update the augmented imu states and corresponding cam states.
  auto imu_state_iter = state_server.imu_states_augment.begin();
  for (int i = 0; i < state_server.imu_states_augment.size();
      ++i, ++imu_state_iter) {
    // update augmented imu state
    const VectorXd& delta_x_aug = delta_x.segment<6>(LEG_DIM+i*6);

    Sophus::SO3d R_temp = Sophus::SO3d::exp(delta_x_aug.head<3>());
    if (use_larvio_flag || use_left_perturbation_flag)
    {
      imu_state_iter->second.orientation = R_temp.matrix() * imu_state_iter->second.orientation;
    }
    else 
    {
      imu_state_iter->second.orientation = imu_state_iter->second.orientation * R_temp.matrix();
    }


    imu_state_iter->second.position += delta_x_aug.tail<3>();
    // update corresponding cam state
    Matrix3d R_c2b =
      state_server.imu_state.R_imu_cam0.transpose();
    const Vector3d& t_c_b =
      state_server.imu_state.t_cam0_imu;
    Matrix3d R_b2w = imu_state_iter->second.orientation;

    Matrix3d R_c2w = R_b2w * R_c2b;
    imu_state_iter->second.orientation_cam = R_c2w;
    imu_state_iter->second.position_cam =
        imu_state_iter->second.position + R_b2w*t_c_b;
  }

}

} // namespace orcvio

