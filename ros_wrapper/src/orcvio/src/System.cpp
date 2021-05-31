//
// Managing the image processer and the estimator.
//

#include <sensor_msgs/PointCloud2.h>

#include <System.h>

#include <iostream>

#include <rosbag/bag.h>
#include <cv_bridge/cv_bridge.h>

#include <geometry_msgs/TransformStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_eigen/tf2_eigen.h>

#include <eigen_conversions/eigen_msg.h>
#include <tf_conversions/tf_eigen.h>

#include <opencv2/core/utility.hpp>

using namespace std;
using namespace cv;
using namespace Eigen;

namespace orcvio {


System::System(ros::NodeHandle& n) : nh(n) {
    
}


System::~System() {
    // Clear buffer
    imu_msg_buffer.clear();
    img_msg_buffer.clear();
    // close file 
    fStateToSave.close();
}


// Load parameters from launch file
bool System::loadParameters() {

    ROS_INFO("System: Start loading ROS parameters...");

    nh.getParam("result_file", result_file);
    first_pose_flag = false; 

    nh.param<std::string>("output_dir_traj", output_dir_traj, "./cache");

    // Configuration file path.
    // template config file 
    nh.getParam("config_file_template", config_file_template);
    FileStorage fs_temp(config_file_template, FileStorage::READ);

    load_groundtruth_flag = 0; 
    nh.getParam("load_groundtruth_flag", load_groundtruth_flag);

    std::string output_dir;
    fs_temp["output_dir"] >> output_dir;
    
    int if_FEJ;
    fs_temp["if_FEJ"] >> if_FEJ;    
    int estimate_extrin;
    fs_temp["estimate_extrin"] >> estimate_extrin;
    int estimate_td;
    fs_temp["estimate_td"] >> estimate_td;
    int calib_imu_instrinsic;
    fs_temp["calib_imu_instrinsic"] >> calib_imu_instrinsic;

    std::string camera_model;
    fs_temp["camera_model"] >> camera_model;
    std::string distortion_model;
    fs_temp["distortion_model"] >> distortion_model;
    int resolution_width;
    fs_temp["resolution_width"] >> resolution_width;
    int resolution_height;
    fs_temp["resolution_height"] >> resolution_height;

    vector<double> intrinsics(4);
    cv::FileNode n_instrin = fs_temp["intrinsics"];
    intrinsics[0] = static_cast<double>(n_instrin["fx"]);
    intrinsics[1] = static_cast<double>(n_instrin["fy"]);
    intrinsics[2] = static_cast<double>(n_instrin["cx"]);
    intrinsics[3] = static_cast<double>(n_instrin["cy"]);

    std::vector<double> distortion_coeffs(4);
    n_instrin = fs_temp["distortion_coeffs"];
    distortion_coeffs[0] = static_cast<double>(n_instrin["k1"]);
    distortion_coeffs[1] = static_cast<double>(n_instrin["k2"]);
    distortion_coeffs[2] = static_cast<double>(n_instrin["p1"]);
    distortion_coeffs[3] = static_cast<double>(n_instrin["p2"]);
    
    Mat T_cam_imu;
    fs_temp["T_cam_imu"] >> T_cam_imu; 
    double td;
    fs_temp["td"] >> td;

    int pyramid_levels;
    fs_temp["pyramid_levels"] >> pyramid_levels;
    int patch_size;
    fs_temp["patch_size"] >> patch_size;
    int fast_threshold;
    fs_temp["fast_threshold"] >> fast_threshold;
    int max_iteration;
    fs_temp["max_iteration"] >> max_iteration;
    double track_precision;
    fs_temp["track_precision"] >> track_precision;
    int ransac_threshold;
    fs_temp["ransac_threshold"] >> ransac_threshold;
    int max_features_num;
    fs_temp["max_features_num"] >> max_features_num;
    int min_distance;
    fs_temp["min_distance"] >> min_distance;
    int flag_equalize;
    fs_temp["flag_equalize"] >> flag_equalize;
    int pub_frequency;
    fs_temp["pub_frequency"] >> pub_frequency;
    
    int sw_size;
    fs_temp["sw_size"] >> sw_size;

    double position_std_threshold;
    fs_temp["position_std_threshold"] >> position_std_threshold;
    double rotation_threshold;
    fs_temp["rotation_threshold"] >> rotation_threshold;
    double translation_threshold;
    fs_temp["translation_threshold"] >> translation_threshold;
    double tracking_rate_threshold;
    fs_temp["tracking_rate_threshold"] >> tracking_rate_threshold;

    int least_observation_number;
    fs_temp["least_observation_number"] >> least_observation_number;
    int max_track_len;
    fs_temp["max_track_len"] >> max_track_len;
    double feature_translation_threshold;
    fs_temp["feature_translation_threshold"] >> feature_translation_threshold;

    double noise_gyro;
    fs_temp["noise_gyro"] >> noise_gyro;
    double noise_acc;
    fs_temp["noise_acc"] >> noise_acc;
    double noise_gyro_bias;
    fs_temp["noise_gyro_bias"] >> noise_gyro_bias;
    double noise_acc_bias;
    fs_temp["noise_acc_bias"] >> noise_acc_bias;
    double noise_feature;
    fs_temp["noise_feature"] >> noise_feature;

    double initial_covariance_orientation;
    fs_temp["initial_covariance_orientation"] >> initial_covariance_orientation;
    double initial_covariance_velocity;
    fs_temp["initial_covariance_velocity"] >> initial_covariance_velocity;
    double initial_covariance_position;
    fs_temp["initial_covariance_position"] >> initial_covariance_position;
    double initial_covariance_gyro_bias;
    fs_temp["initial_covariance_gyro_bias"] >> initial_covariance_gyro_bias;
    double initial_covariance_acc_bias;
    fs_temp["initial_covariance_acc_bias"] >> initial_covariance_acc_bias;
    double initial_covariance_extrin_rot;
    fs_temp["initial_covariance_extrin_rot"] >> initial_covariance_extrin_rot;
    double initial_covariance_extrin_trans;
    fs_temp["initial_covariance_extrin_trans"] >> initial_covariance_extrin_trans;

    double reset_fej_threshold;
    fs_temp["reset_fej_threshold"] >> reset_fej_threshold;

    int if_ZUPT_valid;
    fs_temp["if_ZUPT_valid"] >> if_ZUPT_valid;
    int if_use_feature_zupt_flag;
    fs_temp["if_use_feature_zupt_flag"] >> if_use_feature_zupt_flag;
    double zupt_max_feature_dis;
    fs_temp["zupt_max_feature_dis"] >> zupt_max_feature_dis;
    double zupt_noise_v;
    fs_temp["zupt_noise_v"] >> zupt_noise_v;
    double zupt_noise_p;
    fs_temp["zupt_noise_p"] >> zupt_noise_p;
    double zupt_noise_q;
    fs_temp["zupt_noise_q"] >> zupt_noise_q;

    double static_duration;
    fs_temp["static_duration"] >> static_duration;

    double imu_rate;
    fs_temp["imu_rate"] >> imu_rate;
    double img_rate;
    fs_temp["img_rate"] >> img_rate;

    int max_features_in_one_grid;
    fs_temp["max_features_in_one_grid"] >> max_features_in_one_grid;
    int aug_grid_rows;
    fs_temp["aug_grid_rows"] >> aug_grid_rows;
    int aug_grid_cols;
    fs_temp["aug_grid_cols"] >> aug_grid_cols;
    int feature_idp_dim;
    fs_temp["feature_idp_dim"] >> feature_idp_dim;

    int use_schmidt;
    fs_temp["use_schmidt"] >> use_schmidt;
    int use_left_perturbation_flag;
    fs_temp["use_left_perturbation_flag"] >> use_left_perturbation_flag;
    int use_closed_form_cov_prop_flag;
    fs_temp["use_closed_form_cov_prop_flag"] >> use_closed_form_cov_prop_flag;
    int use_larvio_flag;
    fs_temp["use_larvio_flag"] >> use_larvio_flag;
    int discard_large_update_flag;
    fs_temp["discard_large_update_flag"] >> discard_large_update_flag;

    double chi_square_threshold_feat;
    fs_temp["chi_square_threshold_feat"] >> chi_square_threshold_feat;

    double feature_cost_threshold;
    fs_temp["feature_cost_threshold"] >> feature_cost_threshold;

    double init_final_dist_threshold;
    fs_temp["init_final_dist_threshold"] >> init_final_dist_threshold;

    // for object feature 
    fs_temp["use_object_residual_update_cam_pose_flag"] >> use_object_residual_update_cam_pose_flag;

    // for kitti 
    int initial_use_gt;
    fs_temp["initial_use_gt"] >> initial_use_gt;
    double initial_state_time;
    Mat initial_vel;
    Mat initial_pos;
    Mat initial_quat;
    Mat initial_ba;
    Mat initial_bg;
    if (initial_use_gt)
    {
        fs_temp["initial_state_time"] >> initial_state_time;
        fs_temp["initial_vel"] >> initial_vel; 
        fs_temp["initial_pos"] >> initial_pos; 
        fs_temp["initial_quat"] >> initial_quat; 
        fs_temp["initial_ba"] >> initial_ba;   
        fs_temp["initial_bg"] >> initial_bg;     
    }
    int prediction_only_flag = 0;
    fs_temp["prediction_only_flag"] >> prediction_only_flag;

    ROS_INFO("System: Finish loading template config...");

    // config file to be used for orcvio 
    nh.getParam("config_file", config_file);
    FileStorage fs(config_file, FileStorage::WRITE);
    
    // modify the flags we want to compare 
    // comment all of these below if not in comparison mode! 
    
    int evaluation_mode_flag = 0;
    nh.getParam("evaluation_mode_flag", evaluation_mode_flag);

    if (evaluation_mode_flag)
    {

        std::cout << "in evaluation mode" << std::endl; 
        // nh.getParam("max_features_in_one_grid", max_features_in_one_grid);
        // nh.getParam("use_larvio_flag", use_larvio_flag);
        // nh.getParam("use_left_perturbation_flag", use_left_perturbation_flag);

    }

    fs << "output_dir" << output_dir;

    fs << "if_FEJ" << if_FEJ;
    fs << "estimate_extrin" << estimate_extrin;
    fs << "estimate_td" << estimate_td;
    fs << "calib_imu_instrinsic" << calib_imu_instrinsic;

    fs << "camera_model" << camera_model;
    fs << "distortion_model" << distortion_model;
    fs << "resolution_width" << resolution_width;
    fs << "resolution_height" << resolution_height;

    // write structure 
    // cannot use fs << "intrinsics:" << "{";  !!!
    fs << "intrinsics" << "{";    
    fs << "fx" << intrinsics[0];  
    fs << "fy" << intrinsics[1];  
    fs << "cx" << intrinsics[2];  
    fs << "cy" << intrinsics[3];  
    fs << "}"; 

    // write structure 
    fs << "distortion_coeffs" << "{";    
    fs << "k1" << distortion_coeffs[0];  
    fs << "k2" << distortion_coeffs[1];  
    fs << "p1" << distortion_coeffs[2];  
    fs << "p2" << distortion_coeffs[3];  
    fs << "}"; 

    fs << "T_cam_imu" << T_cam_imu;
    fs << "td" << td;

    fs << "pyramid_levels" << pyramid_levels;
    fs << "patch_size" << patch_size;
    fs << "fast_threshold" << fast_threshold;
    fs << "max_iteration" << max_iteration;
    fs << "track_precision" << track_precision;
    fs << "ransac_threshold" << ransac_threshold;
    fs << "max_features_num" << max_features_num;
    fs << "min_distance" << min_distance;
    fs << "flag_equalize" << flag_equalize;
    fs << "pub_frequency" << pub_frequency;

    fs << "sw_size" << sw_size;

    fs << "position_std_threshold" << position_std_threshold;
    fs << "rotation_threshold" << rotation_threshold;
    fs << "translation_threshold" << translation_threshold;
    fs << "tracking_rate_threshold" << tracking_rate_threshold;

    fs << "least_observation_number" << least_observation_number;
    fs << "max_track_len" << max_track_len;
    fs << "feature_translation_threshold" << feature_translation_threshold;

    fs << "noise_gyro" << noise_gyro;
    fs << "noise_acc" << noise_acc;
    fs << "noise_gyro_bias" << noise_gyro_bias;
    fs << "noise_acc_bias" << noise_acc_bias;
    fs << "noise_feature" << noise_feature;

    fs << "initial_covariance_orientation" << initial_covariance_orientation;
    fs << "initial_covariance_velocity" << initial_covariance_velocity;
    fs << "initial_covariance_position" << initial_covariance_position;
    fs << "initial_covariance_gyro_bias" << initial_covariance_gyro_bias;
    fs << "initial_covariance_acc_bias" << initial_covariance_acc_bias;
    fs << "initial_covariance_extrin_rot" << initial_covariance_extrin_rot;
    fs << "initial_covariance_extrin_trans" << initial_covariance_extrin_trans;

    fs << "reset_fej_threshold" << reset_fej_threshold;

    fs << "if_ZUPT_valid" << if_ZUPT_valid;
    fs << "if_use_feature_zupt_flag" << if_use_feature_zupt_flag;
    fs << "zupt_max_feature_dis" << zupt_max_feature_dis;
    fs << "zupt_noise_v" << zupt_noise_v;
    fs << "zupt_noise_p" << zupt_noise_p;
    fs << "zupt_noise_q" << zupt_noise_q;

    fs << "static_duration" << static_duration;

    fs << "imu_rate" << imu_rate;
    fs << "img_rate" << img_rate;

    fs << "max_features_in_one_grid" << max_features_in_one_grid;
    fs << "aug_grid_rows" << aug_grid_rows;
    fs << "aug_grid_cols" << aug_grid_cols;
    fs << "feature_idp_dim" << feature_idp_dim;

    fs << "use_schmidt" << use_schmidt;
    fs << "use_left_perturbation_flag" << use_left_perturbation_flag;
    fs << "use_closed_form_cov_prop_flag" << use_closed_form_cov_prop_flag;
    fs << "use_larvio_flag" << use_larvio_flag;
    fs << "discard_large_update_flag" << discard_large_update_flag;
    
    fs << "chi_square_threshold_feat" << chi_square_threshold_feat;
    fs << "feature_cost_threshold" << feature_cost_threshold;
    fs << "init_final_dist_threshold" << init_final_dist_threshold;
    
    // for object feature 
    fs << "use_object_residual_update_cam_pose_flag" << use_object_residual_update_cam_pose_flag;

    // for kitti
    fs << "initial_use_gt" << initial_use_gt;
    if (initial_use_gt)
    {
        fs << "initial_state_time" << initial_state_time;
        fs << "initial_vel" << initial_vel;
        fs << "initial_pos" << initial_pos;
        fs << "initial_quat" << initial_quat;
        fs << "initial_ba" << initial_ba;
        fs << "initial_bg" << initial_bg;
    }
    fs << "prediction_only_flag" << prediction_only_flag;

    fs.release();

    ROS_INFO("System: Finish setting config...");

    // Imu and img synchronized threshold.
    // double imu_rate;
    nh.param<double>("imu_rate", imu_rate, 200.0);
    imu_img_timeTh = 1/(2*imu_rate);

    summed_rmse_ori = 0.0;
    summed_rmse_pos = 0.0;
    summed_nees_ori = 0.0;
    summed_nees_pos = 0.0;
    summed_number = 0;

    return true;
}


// Subscribe image and imu msgs.
bool System::createRosIO() {
    // Subscribe imu msg.
    imu_sub = nh.subscribe("imu", 5000, &System::imuCallback, this);

    // Subscribe image msg.
    img_sub = nh.subscribe("cam0_image", 50, &System::imageCallback, this);

    if (use_object_residual_update_cam_pose_flag)
    {
        // Subscribe to object lm status service 
        object_lm_results_client = nh.serviceClient<orcvio_ros_msgs::ObjectLMResults>("/orcvio/get_object_lm_results");
        ROS_INFO("Calling service: %s", object_lm_results_client.getService().c_str());
    }

    // Advertise processed image msg.
    image_transport::ImageTransport it(nh);
    vis_img_pub = it.advertise("visualization_image", 1);

    // Advertise odometry msg.
    odom_pub = nh.advertise<nav_msgs::Odometry>("odom", 10);

    poseout_pub = nh.advertise<geometry_msgs::PoseStamped>("poseout", 2);
    ROS_INFO("Publishing: %s", poseout_pub.getTopic().c_str());

    // Advertise point cloud msg.
    stable_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "stable_feature_point_cloud", 1);
    active_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "active_feature_point_cloud", 1);
    msckf_feature_pub = nh.advertise<sensor_msgs::PointCloud2>(
            "msckf_feature_point_cloud", 1);

    // Advertise path msg.
    path_pub = nh.advertise<nav_msgs::Path>("path", 10);

    nh.param<string>("fixed_frame_id", fixed_frame_id, "world");
    nh.param<string>("child_frame_id", child_frame_id, "robot");

    stable_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    stable_feature_msg_ptr->header.frame_id = fixed_frame_id;
    stable_feature_msg_ptr->height = 1;

    fStateToSave.open((output_dir_traj+"/stamped_traj_estimate.txt").c_str(), std::ofstream::trunc);

    // publish tf 
    nh.param<bool>("publish_tf_flag", publish_tf_flag, true);
    tf2_listener.reset(new tf2_ros::TransformListener(tf2_buffer));

    return true;
}


// Initializing the system.
bool System::initialize() {
    // Load necessary parameters
    if (!loadParameters())
        return false;
    ROS_INFO("System: Finish loading ROS parameters...");

    if (load_groundtruth_flag)
    {

        // load groundtruth
        std::string path_to_gt;
        nh.param<std::string>("path_gt", path_to_gt, "");
        DatasetReader::load_gt_file(path_to_gt, gt_states);

    }

    // Set pointers of image processer and estimator.
    ImgProcesser.reset(new ImageProcessor(config_file));
    Estimator.reset(new OrcVIO(config_file));

    // Initialize image processer and estimator.
    if (!ImgProcesser->initialize()) {
        ROS_WARN("Image Processer initialization failed!");
        return false;
    }
    if (!Estimator->initialize()) {
        ROS_WARN("Estimator initialization failed!");
        return false;
    }

    // Try subscribing msgs
    if (!createRosIO())
        return false;
    ROS_INFO("System Manager: Finish creating ROS IO...");

    return true;
}


// Push imu msg into the buffer.
void System::imuCallback(const sensor_msgs::ImuConstPtr& msg) {

    // for debugging 
    // std::cout << "imu cb timestamp: " << msg->header.stamp.toSec() << std::endl; 

    imu_msg_buffer.push_back(ImuData(msg->header.stamp.toSec(),
            msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z,
            msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z));
}


// Process the image and trigger the estimator.
void System::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    // Do nothing if no imu msg is received.
    if (imu_msg_buffer.empty())
        return;

    cv_bridge::CvImageConstPtr cvCPtr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::MONO8);
    orcvio::ImageDataPtr msgPtr(new ImgData);
    msgPtr->timeStampToSec = cvCPtr->header.stamp.toSec();
    msgPtr->image = cvCPtr->image.clone();
    std_msgs::Header header = cvCPtr->header;
    camera_frame_id = cvCPtr->header.frame_id;

    // for debugging 
    // std::cout << std::fixed;
    // std::cout << std::setprecision(9);
    // std::cout << "img cb timestamp: " << msgPtr->timeStampToSec << std::endl; 
    // exit(0);

    // Decide if use img msg in buffer.
    bool bUseBuff = false;
    if (!imu_msg_buffer.empty() ||
        (imu_msg_buffer.end()-1)->timeStampToSec-msgPtr->timeStampToSec<-imu_img_timeTh) {
        img_msg_buffer.push_back(msgPtr);
        header_buffer.push_back(header);
    }
    if (!img_msg_buffer.empty()) {
        if ((imu_msg_buffer.end()-1)->timeStampToSec-(*(img_msg_buffer.begin()))->timeStampToSec<-imu_img_timeTh)
            return;
        bUseBuff = true;
    }

    if (!bUseBuff) {
        MonoCameraMeasurementPtr features = new MonoCameraMeasurement;

        // Process image to get feature measurement.
        // return false if no feature 
        bool bProcess = ImgProcesser->processImage(msgPtr, imu_msg_buffer, features);

        // Filtering if get processed feature.
        // return false if feature update not begin 
        bool bPubOdo = false;
        if (bProcess) {
            bPubOdo = Estimator->processFeatures(features, imu_msg_buffer);
        }

        if (use_object_residual_update_cam_pose_flag)
        {
            // add object update service request 
            processObjects(header);
        }

        // Publish msgs if necessary
        if (bProcess) {
            cv_bridge::CvImage _image(header, "bgr8", ImgProcesser->getVisualImg());
            vis_img_pub.publish(_image.toImageMsg());
        }
        if (bPubOdo) {
            publishVIO(header.stamp);

            if (load_groundtruth_flag)
                publishGroundtruth(header.stamp);

        }

        delete features;

        return;
    } else {
        // Loop for using all the img in the buffer that satisfy the condition.
        int counter = 0;
        for (int i = 0; i < img_msg_buffer.size(); ++i) {
            // Break the loop if imu data is not enough
            if ((imu_msg_buffer.end()-1)->timeStampToSec-img_msg_buffer[i]->timeStampToSec<-imu_img_timeTh)
                break;

            MonoCameraMeasurementPtr features = new MonoCameraMeasurement;

            // Process image to get feature measurement.
            bool bProcess = ImgProcesser->processImage(img_msg_buffer[i], imu_msg_buffer, features);

            // Filtering if get processed feature.
            bool bPubOdo = false;
            if (bProcess) {
                bPubOdo = Estimator->processFeatures(features, imu_msg_buffer);
            }

            if (use_object_residual_update_cam_pose_flag)
            {
                // add object update service request
                processObjects(header);
            }

            // Publish msgs if necessary
            if (bProcess) {
                cv_bridge::CvImage _image(header_buffer[i], "bgr8", ImgProcesser->getVisualImg());
                vis_img_pub.publish(_image.toImageMsg());
            }
            if (bPubOdo) {
                publishVIO(header_buffer[i].stamp);

                if (load_groundtruth_flag)
                    publishGroundtruth(header.stamp);

            }

            delete features;

            counter++;
        }
        img_msg_buffer.erase(img_msg_buffer.begin(), img_msg_buffer.begin()+counter);
        header_buffer.erase(header_buffer.begin(), header_buffer.begin()+counter);
    }
}


// Push imu msg into the buffer.
void System::processObjects(const std_msgs::Header& header) {

    // request object LM message 
    // send a service request to object LM 
    // this is a blocking version of object LM, since 
    // VIO will not publish pose before object LM finishes  
    orcvio_ros_msgs::ObjectLMResults::Request req;
    orcvio_ros_msgs::ObjectLMResults::Response res;

    ros::service::waitForService("/orcvio/get_object_lm_results");

    if (!object_lm_results_client.call(req, res))
    {
        ROS_WARN("[VIO System] Failed to call object lm results server");
    }

    const orcvio_ros_msgs::ObjectLMList* msg = &res.msg;  

    // check if there is no object LM msg 
    if (msg->object_lm_msgs.size() == 0)
        return; 

    // for debugging 
    if (header.stamp.toSec() != msg->header.stamp.toSec())
    {
        std::cout << "VIO timestamp " << header.stamp.toSec() << std::endl; 
        std::cout << "object LM timestamp " << msg->header.stamp.toSec() << std::endl;
    }

    // make sure the object LM result has same timestamp with current VIO 
    assert(header.stamp.toSec() == msg->header.stamp.toSec() && "[VIO System] object lm results server timestamp different from VIO timestamp");

    MatrixXd Hx;
    MatrixXd Hf;
    VectorXd r;

    std::cout << "[Object callback] Received " << msg->object_lm_msgs.size() << " messages" << std::endl; 

    for (const auto & object_lm_msg : msg->object_lm_msgs)
    {
        int object_id = object_lm_msg.object_id;
        std::vector<double> timestamps = object_lm_msg.timestamps;
        Eigen::VectorXd r_j = msgToEigen(object_lm_msg.residual);
        Eigen::MatrixXd Hf_j = msgToEigen(object_lm_msg.jacobian_wrt_object_state);

        std::vector<int> zs_num_wrt_timestamps = object_lm_msg.zs_num_wrt_timestamps; 
        Eigen::MatrixXd jacobian_wrt_sensor_state = msgToEigen(object_lm_msg.jacobian_wrt_sensor_state);

        Eigen::MatrixXd valid_camera_pose_mat = msgToEigen(object_lm_msg.valid_camera_pose_mat);

        // Hx_j is a place holder 
        Eigen::MatrixXd Hx_j;
        bool has_pose_in_window_flag = Estimator->constructObjectResidualJacobians(jacobian_wrt_sensor_state, timestamps, 
                Hx_j, Hf_j, r_j, zs_num_wrt_timestamps, valid_camera_pose_mat);
        
        if (!has_pose_in_window_flag)
        {
            std::cout << "[Object callback] object not in current window" << std::endl; 
            continue;
        } 

        // stack the jacobians, residuals vertically 
        if (r.rows() == 0)
        {
            Hx = Hx_j;
            Hf = Hf_j; 
            r = r_j; 
        }
        else 
        {

            Hx.conservativeResize(Hx.rows() + Hx_j.rows(), Hx.cols());
            Hx.bottomRows(Hx_j.rows()) = Hx_j;

            Hf.conservativeResize(Hf.rows() + Hf_j.rows(), Hf.cols());
            Hf.bottomRows(Hf_j.rows()) = Hf_j;

            r.conservativeResize(r.rows() + r_j.rows(), r.cols());
            r.bottomRows(r_j.rows()) = r_j;

        }

    }

    Estimator->removeLostObjects(Hx, Hf, r);

}

Eigen::MatrixXd System::msgToEigen(const std_msgs::Float64MultiArray& msg)
{
	double dstride0 = msg.layout.dim[0].stride;
	double dstride1 = msg.layout.dim[1].stride;
	double h = msg.layout.dim[0].size;
	double w = msg.layout.dim[1].size;

	// Below are a few basic Eigen demos:
	std::vector<double> data = msg.data;
	Eigen::Map<Eigen::MatrixXd> mat(data.data(), h, w);
	// std::cout << "I received = " << std::endl << mat << std::endl;
	
	return mat;
}

const std::string System::find_tf_tree_root(
    const std::string& frame_id, const ros::Time& time)
{
    std::string cursor = frame_id;
    std::string parent;
    while (tf2_buffer._getParent(cursor, time, parent))
        cursor = parent;
    // ROS_WARN_STREAM("Found root : " << cursor);
    return cursor;
}

// Publish informations of VIO, including odometry, path, points cloud and whatever needed.
void System::publishVIO(const ros::Time& time) {

    // construct odometry msg

    odom_msg.header.stamp = time;
    odom_msg.header.frame_id = fixed_frame_id;
    odom_msg.child_frame_id = camera_frame_id;
    Eigen::Isometry3d T_b_w = Estimator->getTbw();
    Eigen::Vector3d body_velocity = Estimator->getVel();
    Matrix<double, 6, 6> P_body_pose = Estimator->getPpose();
    Matrix3d P_body_vel = Estimator->getPvel();

    // use IMU pose 
    // tf::poseEigenToMsg(T_b_w, odom_msg.pose.pose);
    // use camera pose 
    Eigen::Isometry3d T_c_w = Estimator->getTcw();
    tf::poseEigenToMsg(T_c_w, odom_msg.pose.pose);

    tf::vectorEigenToMsg(body_velocity, odom_msg.twist.twist.linear);
    for (int i = 0; i < 6; ++i)
        for (int j = 0; j < 6; ++j)
            odom_msg.pose.covariance[6*i+j] = P_body_pose(i, j);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            odom_msg.twist.covariance[i*6+j] = P_body_vel(i, j);

    // construct path msg
    path_msg.header.stamp = time;
    path_msg.header.frame_id = fixed_frame_id;
    geometry_msgs::PoseStamped curr_path;
    curr_path.header.stamp = time;
    curr_path.header.frame_id = fixed_frame_id;
    tf::poseEigenToMsg(T_b_w, curr_path.pose);
    path_msg.poses.push_back(curr_path);

    // construct pose msg
    // from imu to global 
    // this is used by object mapping 
    geometry_msgs::PoseStamped wTi_msg;
    wTi_msg.header.stamp = time;
    wTi_msg.header.frame_id = camera_frame_id;
    wTi_msg.header.seq = poses_seq_out;
    // use IMU pose 
    tf::poseEigenToMsg(T_b_w, wTi_msg.pose);

    // construct point cloud msg
    // Publish the 3D positions of the features.
    // Including stable and active ones.
    // --Stable features
    std::map<orcvio::FeatureIDType,Eigen::Vector3d> StableMapPoints;
    Estimator->getStableMapPointPositions(StableMapPoints);
    for (const auto& item : StableMapPoints) {
        const auto& feature_position = item.second;
        stable_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    stable_feature_msg_ptr->width = stable_feature_msg_ptr->points.size();
    // --Active features
    active_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    active_feature_msg_ptr->header.frame_id = fixed_frame_id;
    active_feature_msg_ptr->height = 1;
    std::map<orcvio::FeatureIDType, Eigen::Vector3d> ActiveMapPoints;
    Estimator->getActiveMapPointPositions(ActiveMapPoints);
    for (const auto& item : ActiveMapPoints) {
        const auto& feature_position = item.second;
        active_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    active_feature_msg_ptr->width = active_feature_msg_ptr->points.size();

    // publish msckf features
    msckf_feature_msg_ptr.reset(
        new pcl::PointCloud<pcl::PointXYZ>());
    msckf_feature_msg_ptr->header.frame_id = fixed_frame_id;
    msckf_feature_msg_ptr->height = 1;
    std::map<orcvio::FeatureIDType, Eigen::Vector3d> MSCKFPoints;
    Estimator->getMSCKFMapPointPositions(MSCKFPoints);
    for (const auto& item : MSCKFPoints) {
        const auto& feature_position = item.second;
        msckf_feature_msg_ptr->points.push_back(pcl::PointXYZ(
                feature_position(0), feature_position(1), feature_position(2)));
    }
    msckf_feature_msg_ptr->width = msckf_feature_msg_ptr->points.size();

    odom_pub.publish(odom_msg);
    path_pub.publish(path_msg);
    poseout_pub.publish(wTi_msg);

    if (publish_tf_flag) 
    {
        // Subscript notation for transforms, (i)mu, (c)amera, robot-(b)ase, (w)orld
        // wTi is transform that converts from i -> w: p_w = wTi @ p_i
        Eigen::Isometry3d wTc_eigen = T_c_w;
        Eigen::Isometry3d cTb_eigen;
        try
        {
            if (!base_frame_id.size())
                base_frame_id = find_tf_tree_root(camera_frame_id, time);

            geometry_msgs::TransformStamped cTb_msg =
                tf2_buffer.lookupTransform(camera_frame_id, base_frame_id,
                                            time,
                                            /*timeout=*/ros::Duration(0.1));
            cTb_eigen = tf2::transformToEigen(cTb_msg.transform);
        }
        catch (tf2::TransformException &ex) 
        {
            ROS_WARN_STREAM("Unable to get base -> image transform." <<
                            ex.what() << ". Assuming identity. ");
            // Not found
            // Set to identity transform
            cTb_eigen = Eigen::Isometry3d::Identity();
        }
        Eigen::Isometry3d wTb_eigen = wTc_eigen * cTb_eigen;
        geometry_msgs::TransformStamped transform = tf2::eigenToTransform(wTb_eigen);
        transform.header.seq = poses_seq_out;
        transform.header.stamp = time;
        transform.header.frame_id = fixed_frame_id; // (w)orld
        transform.child_frame_id = base_frame_id; // robot-(b)ase
        // http://wiki.ros.org/tf2/Tutorials/Writing%20a%20tf2%20broadcaster%20%28C%2B%2B%29
        // If the position of child origin in world frame acts as translation, then
        // the transform transforms from child_frame_id -> header.frame_id
        // unlike what is documented in geometry_msgs/TransformStamped.
        // http://docs.ros.org/en/api/geometry_msgs/html/msg/TransformStamped.html
        poseout_tf2_broadcaster.sendTransform(transform);
    }

    // Move them forward in time
    poses_seq_out++;

    stable_feature_pub.publish(stable_feature_msg_ptr);
    active_feature_pub.publish(active_feature_msg_ptr);
    msckf_feature_pub.publish(msckf_feature_msg_ptr);

    // save the pose to txt for trajectory evaluation 
    // timestamp tx ty tz qx qy qz qw
    fStateToSave << std::fixed << std::setprecision(3) << curr_path.header.stamp.toSec();
    fStateToSave << " "
        << curr_path.pose.position.x << " " << curr_path.pose.position.y << " " << curr_path.pose.position.z << " "
        << curr_path.pose.orientation.x << " " << curr_path.pose.orientation.y << " " << curr_path.pose.orientation.z << " " << curr_path.pose.orientation.w << std::endl; 

    // fStateToSave << curr_path.header.stamp.toSec() << " "
    //     << curr_path.pose.position.x << " " << curr_path.pose.position.y << " " << curr_path.pose.position.z << " "
    //     << curr_path.pose.orientation.x << " " << curr_path.pose.orientation.y << " " << curr_path.pose.orientation.z << " " << curr_path.pose.orientation.w << std::endl; 

}

void System::publishGroundtruth(const ros::Time& time) {

    double timestamp = time.toSec();
    // Our groundtruth state
    Eigen::Matrix<double,17,1> state_gt;

    // Check that we have the timestamp in our GT file [time(sec),q_GtoI,p_IinG,v_IinG,b_gyro,b_accel]
    if(!DatasetReader::get_gt_state(timestamp, state_gt, gt_states)) {
        return;
    }

    Eigen::Vector4d q_gt;
    Eigen::Vector3d p_gt;

    q_gt << state_gt(1,0),state_gt(2,0),state_gt(3,0),state_gt(4,0);
    p_gt << state_gt(5,0), state_gt(6,0), state_gt(7,0);

    Eigen::Isometry3d T_b_w = Estimator->getTbw();

    if (!first_pose_flag)
    {
        // load the first pose 
        Eigen::Matrix4d first_pose_gt;
        first_pose_gt.block<3, 3>(0, 0) = quaternionToRotation(q_gt);
        first_pose_gt.block<3, 1>(0, 3) = p_gt;

        T_from_est_to_gt = first_pose_gt * T_b_w.inverse().matrix();

        first_pose_flag = true;
    }

    Eigen::Matrix4d T_est_corrected = T_from_est_to_gt * T_b_w.matrix();
    Eigen::Matrix3d wRi = T_est_corrected.block<3, 3>(0, 0);
    Eigen::Vector3d wPi = T_est_corrected.block<3, 1>(0, 3);

    // Difference between positions
    double dx = wPi(0)-p_gt(0);
    double dy = wPi(1)-p_gt(1);
    double dz = wPi(2)-p_gt(2);
    double rmse_pos = std::sqrt(dx*dx+dy*dy+dz*dz);

    // Quaternion error
    Eigen::Matrix<double,4,1> quat_st, quat_diff;

    quat_st = rotationToQuaternion(wRi);   
    Eigen::Vector4d quat_gt_inv = inverseQuaternion(q_gt);
    quat_diff = quaternionMultiplication(quat_st, quat_gt_inv);
    double rmse_ori = (180/M_PI)*2*quat_diff.block(0,0,3,1).norm();

    // Update our average variables
    summed_rmse_ori += rmse_ori;
    summed_rmse_pos += rmse_pos;
    summed_number++;

    FILE* fp = fopen(result_file.c_str(), "w");
    fprintf(fp, "%f %f\n", summed_rmse_ori/summed_number, summed_rmse_pos/summed_number);
    fclose(fp);

}

} // end namespace orcvio
