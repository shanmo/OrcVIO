#include <sstream>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <set>

#include <ObjectInitNode.h>

using Eigen::Matrix3d;
using Eigen::Vector4d;
using Eigen::VectorXd;

using namespace boost::filesystem;

namespace orcvio
{

    ObjectInitNode::~ObjectInitNode()
    {
        fBboxToSave.close();
    }

    ObjectInitNode::ObjectInitNode(ros::NodeHandle &nh)
    {

        // quadrics publishing
        pub_quadrics = nh.advertise<visualization_msgs::MarkerArray>("/orcvio/quadrics", 2);
        ROS_INFO("Publishing: %s", pub_quadrics.getTopic().c_str());

        pub_gt_objects = nh.advertise<visualization_msgs::MarkerArray>("/orcvio/gt_objects", 2);
        ROS_INFO("Publishing: %s", pub_gt_objects.getTopic().c_str());

        // service publishing
        service = nh.advertiseService("/orcvio/get_object_lm_results", &ObjectInitNode::callback_objectLMBlocking, this);
        ROS_INFO("Publishing service: %s", service.getService().c_str());

#ifdef SYNC_WITH_IMG 
        // track image publishing
        nh.param<std::string>("track_image_topic", track_image_topic, "/orcvio/track_image");
#endif 

        nh.param<std::string>("fixed_frame_id", fixed_frame_id, "world");

        // Create subscribers
        nh.param<std::string>("topic_keypoint", topic_keypoint, "/starmap/keypoints");
        nh.param<std::string>("topic_pose", topic_pose, "/unity_ros/husky/TrueState/odom");
        nh.param<std::string>("topic_caminfo", topic_caminfo, "/husky/camera/camera_info");
        
#ifdef SYNC_WITH_IMG 
        // for plotting keypoint tracks 
        nh.param<std::string>("topic_image", topic_image, "/husky/camera/image");
#endif 

        // get the dir path to save object map
        std::string ros_log_dir;
        ros::get_environment_variable(ros_log_dir, "ROS_LOG_DIR");
        nh.param<std::string>("result_dir_path_object_map", result_dir_path_object_map, ros_log_dir);

        if (exists(result_dir_path_object_map))
        {
            directory_iterator end_it;
            directory_iterator it(result_dir_path_object_map.c_str());
            if (it == end_it)
            {
                // this is fine
            }
            else
            {
                ROS_INFO_STREAM("object map path exists and nonempty, delete contents in " << result_dir_path_object_map.c_str());
                // if this dir already exists, then delete all contents inside
                std::string del_cmd = "exec rm -r " + result_dir_path_object_map + "*";
                int tmp = system(del_cmd.c_str());
            }
        }
        else
        {
            // if this dir does not exist, create the dir
            const char *path = result_dir_path_object_map.c_str();
            boost::filesystem::path dir(path);
            if (boost::filesystem::create_directory(dir))
            {
                std::cerr << "Directory Created: " << result_dir_path_object_map << std::endl;
            }
        }

        // get the path to store bounding box info. used in
        // object IOU evaluation for KITTI dataset
        nh.param<std::string>("result_dir_path_2d_bbox", result_dir_path_2d_bbox, "/cache/object_2d_bbox_info.txt");
        fBboxToSave.open(result_dir_path_2d_bbox.c_str(), std::ofstream::trunc);
        if (fBboxToSave.fail())
        {
            std::cout << "Failed to open " << result_dir_path_2d_bbox.c_str() << std::endl;
        }
        else
        {
            std::cout << "Success to open " << result_dir_path_2d_bbox.c_str() << std::endl;
        }

        sub_caminfo = nh.subscribe(topic_caminfo.c_str(), 9999, &ObjectInitNode::callback_caminfo, this);
        sub_gtpose = nh.subscribe(topic_pose.c_str(), 9999, &ObjectInitNode::callback_pose, this);
#ifndef SYNC_WITH_IMG 
        sub_sem = nh.subscribe(topic_keypoint.c_str(), 9999, &ObjectInitNode::callback_sem, this);
#endif 

#ifdef SYNC_WITH_IMG 
        // for plotting
        sub_sem = make_unique<message_filters::Subscriber<starmap_ros_msgs::TrackedBBoxListWithKeypoints>>(nh, topic_keypoint, 1);
        sub_img = make_unique<message_filters::Subscriber<sensor_msgs::Image>>(nh, topic_image, 1);
        namespace sph = std::placeholders; // for _1, _2, ...

        // fixed queue size 
        // const int queue_size = 1; 
        // sub_sem_img = make_unique<message_filters::TimeSynchronizer<sensor_msgs::Image, starmap_ros_msgs::TrackedBBoxListWithKeypoints>>(*sub_img, *sub_sem, queue_size);
        // approximate sync 
        sub_sem_img = make_unique<message_filters::Synchronizer<MySyncPolicy> > (MySyncPolicy(100), *sub_img, *sub_sem);

        sub_sem_img->registerCallback(std::bind(&ObjectInitNode::callback_sem, this, sph::_1, sph::_2));

        image_trans = make_unique<image_transport::ImageTransport>(nh);
        trackImagePublisher = image_trans->advertise(track_image_topic, 10);
#endif 

        // Our camera extrinsics transform used in getting camera pose
        // since VIO only outputs IMU pose but object LM needs camera pose
        // Read in from ROS, and save into our eigen mat
        std::vector<double> matrix_TItoC;
        std::vector<double> matrix_TItoC_default = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
        int i = 0;
        nh.param<std::vector<double>>("T_cam_imu", matrix_TItoC, matrix_TItoC_default);
        // step 1: load T_ItoC, this means from imu to cam
        T_ItoC << matrix_TItoC.at(0), matrix_TItoC.at(1), matrix_TItoC.at(2), matrix_TItoC.at(3),
            matrix_TItoC.at(4), matrix_TItoC.at(5), matrix_TItoC.at(6), matrix_TItoC.at(7),
            matrix_TItoC.at(8), matrix_TItoC.at(9), matrix_TItoC.at(10), matrix_TItoC.at(11),
            matrix_TItoC.at(12), matrix_TItoC.at(13), matrix_TItoC.at(14), matrix_TItoC.at(15);

        // step 2: get T_CtoI
        T_CtoI = T_ItoC.inverse().eval();

        // determine whether we are using the lite version
        // if so, make the mean shape shperes
        nh.param<bool>("use_bbox_only_flag", use_bbox_only_flag, false);

        // #--------------------------------------------------------------------------------------------
        // # different object classes start
        // #--------------------------------------------------------------------------------------------

        XmlRpc::XmlRpcValue object_classes;
        const std::string OBJECT_CLASSES_KEY = "object_classes";
        nh.getParam(OBJECT_CLASSES_KEY, object_classes);
        ROS_ASSERT(object_classes.getType() == XmlRpc::XmlRpcValue::TypeStruct);
        for (XmlRpc::XmlRpcValue::const_iterator key_value = object_classes.begin();
             key_value != object_classes.end();
             ++key_value)
        {

            auto const &class_name = static_cast<const std::string>(key_value->first);
            std::vector<double> object_mean_shape_vec(3);
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/object_mean_shape", object_mean_shape_vec);

            if (use_bbox_only_flag)
            {
                convert_quad_to_sphere(object_mean_shape_vec);
            }

            object_sizes_gt_dict[class_name] = object_mean_shape_vec;

            int kps_num;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/keypoints_num", kps_num);
            std::vector<double> object_keypoints_mean_vec(3 * kps_num);
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/object_keypoints_mean", object_keypoints_mean_vec);
            Eigen::MatrixXd object_mean_shape_mat = Eigen::Map<Eigen::Matrix<double, 3, 1>>(object_mean_shape_vec.data());
            Eigen::MatrixXd object_keypoints_mean_mat = Eigen::Map<Eigen::MatrixXd>(
                object_keypoints_mean_vec.data(), kps_num, 3);
            std::vector<std::string> object_accepted_names;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/aliases", object_accepted_names);
            object_accepted_names.push_back(class_name);
            for (auto const &name : object_accepted_names)
                object_standardized_class_name_[name] = class_name;
            setup_object_feature_initializer(class_name,
                                             object_mean_shape_mat, object_keypoints_mean_mat);

            std::vector<double> marker_color;
            nh.getParam(OBJECT_CLASSES_KEY + "/" + class_name + "/marker_color", marker_color);
            object_marker_colors_[class_name] = marker_color;
        }

        // #--------------------------------------------------------------------------------------------
        // # different object classes end
        // #--------------------------------------------------------------------------------------------

        gt_object_map_saved_flag = false;

        int max_object_feature_track_length;
        int min_object_feature_track_length;
        nh.getParam("max_object_feature_track", max_object_feature_track_length);
        nh.getParam("min_object_feature_track", min_object_feature_track_length);
        set_track_length(min_object_feature_track_length, max_object_feature_track_length);

        // determine which dataset we are using
        nh.param<bool>("use_unity_dataset_flag", use_unity_dataset_flag, false);
        nh.param<bool>("to_color_image_flag", to_color_image_flag, false);

        // load the flag for object LM
        nh.param<bool>("do_fine_tune_object_pose_using_lm", do_fine_tune_object_pose_using_lm, false);

        // load the flag for new bounding box residual
        nh.param<bool>("use_new_bbox_residual_flag", use_new_bbox_residual_flag, false);

        // load the flag for using left or right perturbation
        nh.param<bool>("use_left_perturbation_flag", use_left_perturbation_flag, true);

        // load gt object states
        nh.param<bool>("load_gt_object_info_flag", load_gt_object_info_flag, true);

        if (load_gt_object_info_flag)
        {

            int total_object_num = 0;
            nh.getParam("total_object_num", total_object_num);

            std::vector<int> objects_ids_gt(total_object_num);
            nh.getParam("objects_ids_gt", objects_ids_gt);

            std::vector<std::string> objects_class_gt(total_object_num);
            nh.getParam("objects_class_gt", objects_class_gt);

            std::vector<double> objects_rotation_gt(9 * total_object_num);
            nh.getParam("objects_rotation_gt", objects_rotation_gt);

            std::vector<double> objects_translation_gt(3 * total_object_num);
            nh.getParam("objects_translation_gt", objects_translation_gt);

            std::vector<double> object_position_gt(3);
            std::vector<double> object_rotation_gt(9);

            for (int i = 0; i < total_object_num; i++)
            {

                // All sequence containers in C++ preserve internal order
                int object_id_gt = objects_ids_gt[i];
                std::string object_class_gt = objects_class_gt[i];

                object_position_gt = std::vector<double>(objects_translation_gt.begin() + (i * 3), objects_translation_gt.begin() + ((i + 1) * 3));
                object_rotation_gt = std::vector<double>(objects_rotation_gt.begin() + (i * 9), objects_rotation_gt.begin() + ((i + 1) * 9));

                Eigen::MatrixXd object_position_gt_mat = Eigen::Map<Eigen::Matrix<double, 3, 1>>(object_position_gt.data());
                Eigen::MatrixXd object_rotation_gt_mat = Eigen::Map<Eigen::Matrix<double, 3, 3>>(object_rotation_gt.data());

                // for debugging
                // for (const auto & i : object_rotation_gt)
                //     std::cout << "object_rotation_gt " << i << std::endl;
                // std::cout << "object_rotation_gt_mat " << object_rotation_gt_mat << std::endl;

                add_gt_object_state(object_class_gt, object_id_gt, object_position_gt_mat, object_rotation_gt_mat);
            }

            std::vector<double> first_uav_translation_gt_vec(3);
            nh.getParam("first_uav_translation_gt", first_uav_translation_gt_vec);
            first_uav_translation_gt = Eigen::Map<Eigen::Matrix<double, 3, 1>>(first_uav_translation_gt_vec.data());
        
        }

    } // end of ObjectInitNode

    void ObjectInitNode::add_gt_object_state(const std::string &object_class, const int &object_id, const Eigen::MatrixXd &object_position_gt_mat, const Eigen::MatrixXd &object_rotation_gt_mat)
    {
        object_id_gt_vec.push_back(object_id);
        object_class_gt_vec.push_back(object_class);

        object_position_gt_vec.push_back(object_position_gt_mat);
        object_rotation_gt_vec.push_back(object_rotation_gt_mat);
    }

    void ObjectInitNode::set_track_length(const int &min_object_feature_track_length, const int &max_object_feature_track_length)
    {

        this->max_object_feature_track_length = (unsigned)max_object_feature_track_length;
        this->min_object_feature_track_length = (unsigned)min_object_feature_track_length;
    }

    void ObjectInitNode::setup_object_feature_initializer(const std::string &object_class, const Eigen::Vector3d &object_mean_shape, const Eigen::MatrixX3d &object_keypoints_mean)
    {
        Eigen::Matrix<double, 3, 3> camera_intrinsics;
        cv2eigen(camK, camera_intrinsics);

        auto class_name_iter = object_standardized_class_name_.find(object_class);
        if (class_name_iter == object_standardized_class_name_.end())
        {
            // unknown object class
            ROS_WARN_STREAM("Ignoring unknown object class: " << object_class << ". Known classes : ");
            return;
        }
        auto obj_feat_init_iter = all_objects_feat_init_.find(class_name_iter->second);
        if (obj_feat_init_iter == all_objects_feat_init_.end())
        {
            all_objects_feat_init_[class_name_iter->second] =
                std::make_shared<ObjectFeatureInitializer>(
                    featinit_options, object_mean_shape, object_keypoints_mean,
                    camera_intrinsics);
        }
        else
        {
            obj_feat_init_iter->second.reset(
                new ObjectFeatureInitializer(
                    featinit_options, object_mean_shape, object_keypoints_mean,
                    camera_intrinsics));
        }
    }

    void ObjectInitNode::convert_quad_to_sphere(std::vector<double> &mean_shape)
    {
        // if the mean shape has equal width and length, then do not do conversion
        // if the mean shape has different width and length, then set those to the average
        // do not change height during conversion

        if (mean_shape[0] == mean_shape[1])
            return;
        else
        {
            double avg = (mean_shape[0] + mean_shape[1]) / 2;
            mean_shape[0] = avg;
            mean_shape[1] = avg;
        }
    }

    // this is for general case
    void ObjectInitNode::callback_caminfo(const sensor_msgs::CameraInfoConstPtr &cam_info)
    {
        // for intrinsics
        std::vector<double> cam0_intrinsics_temp(4);
        std::vector<double> cam0_distortion_coeffs_temp(4);

        camK << cam_info->K[0], 0, cam_info->K[2],
            0, cam_info->K[4], cam_info->K[5],
            0, 0, 1;

        camD << cam_info->D[0], cam_info->D[1], cam_info->D[2], cam_info->D[3];

        // for debugging
        // std::cout << camK << std::endl;
        // std::cout << camD << std::endl;

        sub_caminfo.shutdown();
    }

    // this is for kitti odometry rosbag only
    // void ObjectInitNode::callback_caminfo(const sensor_msgs::CameraInfoConstPtr& cam_info)
    // {
    //     // for intrinsics
    //     std::vector<double> cam0_intrinsics_temp(4);
    //     std::vector<double> cam0_distortion_coeffs_temp(4);

    //     camK <<  cam_info->P[0], 0, cam_info->P[2],
    //             0, cam_info->P[5], cam_info->P[6],
    //             0, 0, 1;

    //     camD << 0, 0, 0, 0;

    //     // for debugging
    //     // std::cout << camK << std::endl;
    //     // std::cout << camD << std::endl;

    //     sub_caminfo.shutdown();
    // }

    void ObjectInitNode::callback_pose(const geometry_msgs::PoseStamped::ConstPtr &odom_ptr)
    {

        // convert to quaternion
        Eigen::Quaterniond q1;
        q1.x() = odom_ptr->pose.orientation.x;
        q1.y() = odom_ptr->pose.orientation.y;
        q1.z() = odom_ptr->pose.orientation.z;
        q1.w() = odom_ptr->pose.orientation.w;

        // convert to rotation matrix
        Matrix3d R1;
        R1 = q1.normalized().toRotationMatrix();

        Eigen::Vector3d t1;
        t1 << odom_ptr->pose.position.x, odom_ptr->pose.position.y, odom_ptr->pose.position.z;

        // Get current camera pose
        Eigen::Matrix<double, 3, 3> R_ItoG;
        R_ItoG = R1;

        Eigen::Matrix<double, 3, 1> p_IinG;
        p_IinG(0, 0) = t1(0, 0);
        p_IinG(1, 0) = t1(1, 0);
        p_IinG(2, 0) = t1(2, 0);

        // for debugging 
        // std::cout << "p " << p_IinG << std::endl; 
        // std::cout << "R " << R_ItoG << std::endl; 

        // Append to our map
        clones_imu.insert({odom_ptr->header.stamp.toSec(), FeatureInitializer::ClonePose(R_ItoG, p_IinG)});
        pose_timestamps.push_back(odom_ptr->header.stamp.toSec());
    }

    /**
    * @brief print the lost flag  
    * @param boolean lost flag 
    */
    inline const std::string BoolToString(bool b) { return b ? "true" : "false"; }

    // must use a ConstPtr callback to use zero-copy transport
#ifdef SYNC_WITH_IMG 
    void ObjectInitNode::callback_sem(const sensor_msgs::ImageConstPtr &message,
                                      const starmap_ros_msgs::TrackedBBoxListWithKeypointsConstPtr &bbox_kp_msg)
#endif
#ifndef SYNC_WITH_IMG  
    void ObjectInitNode::callback_sem(const starmap_ros_msgs::TrackedBBoxListWithKeypointsConstPtr& bbox_kp_msg) 
#endif 
    {

        // always record the timestamps even if there is no bounding box
        fBboxToSave << std::fixed << std::setprecision(3) << bbox_kp_msg->header.stamp.toSec() << std::endl;

        if (cv::countNonZero(camK) == 0)
        {
            ROS_WARN_STREAM("camK is still 0. Dropping this stream of bounding box messages");
            return;
        }

#ifdef SYNC_WITH_IMG 
        const bool show_track_image_flag = true; 
        cv::Mat track_image;
        if (show_track_image_flag)
        {
            // display tracking results
            child_frame_id = message->header.frame_id;
            cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message);
            track_image = img->image;
            if (to_color_image_flag)
            {
                // convert to rgb image if necessary
                cv::cvtColor(track_image, track_image, cv::COLOR_GRAY2RGB);
            }
        }
#endif 

        // we only use 1 camera for semantic observations
        int cam_id = 0;

        // set the header for object lm results msg
        object_lm_list_msg.header = bbox_kp_msg->header;

        for (const auto &bbox_with_kp : bbox_kp_msg->bounding_boxes)
        {

            if (!use_bbox_only_flag)
            {
                // for debugging
                // std::cout << "keypoints size " << bbox_with_kp.keypoints.size() << std::endl;
                if (bbox_with_kp.keypoints.size() < 1)
                    continue;
            }

            const auto &bbox = bbox_with_kp.bbox;

            // check whether the object has already been optimized 
            // if we encounter the same object again, do not track and 
            // do not do optimization for the second time 
            // for now this is mainly used in the Unity dataset 
            if (all_object_states_dict.find(bbox.id) != all_object_states_dict.end())
            {
                // object exists in the current map, skip tracking and optimization 
                continue; 
            }

            // check whether the object class is valid
            std::string standard_object_class;
            auto class_name_iter = object_standardized_class_name_.find(bbox.Class);
            if (class_name_iter == object_standardized_class_name_.end())
            {
                // unknown object class
                ROS_WARN_STREAM("Ignoring unknown object class: " << bbox.Class);
                continue;
            }
            else
            {
                standard_object_class = class_name_iter->second;
                // for debugging 
                // ROS_WARN_STREAM("Detected object class: " << bbox.Class);
                // ROS_WARN_STREAM("Standard object class: " << standard_object_class);
            }

            std::shared_ptr<ObjectFeature> obj_obs_ptr;

            // for debugging
            // std::cout << "bbox id " << bbox.id << std::endl;

            // save bbox info. for 3D IOU evaluation in KITTI
            // format [timestamp, object_id, *bbox] where bbox format is
            // x min, y min, x max, y max
            fBboxToSave << std::fixed << std::setprecision(3) << bbox_kp_msg->header.stamp.toSec() << " " << bbox.id << " " << bbox.xmin << " " << bbox.ymin << " " << bbox.xmax << " " << bbox.ymax << std::endl;
            // fBboxToSave << std::fixed << std::setprecision(3) << message->header.stamp.toSec() << " " << bbox.id << " " << bbox.xmin << " " << bbox.ymin << " " << bbox.xmax << " " << bbox.ymax << std::endl;

            // check if the object appears the first time
            if (object_obs_dict.find(bbox.id) == object_obs_dict.end())
            {
                obj_obs_ptr.reset(new ObjectFeature(bbox.id, standard_object_class));

                // for debugging
                // std::cout << "New object id " << obj_obs_ptr->object_id << std::endl;
                // std::cout << "New object class " << obj_obs_ptr->object_class << std::endl;

                // insert new object observations pointer
                object_obs_dict[obj_obs_ptr->object_id] = obj_obs_ptr;
            }
            else
            {
                obj_obs_ptr = object_obs_dict.at(bbox.id);
            }

            // for debugging
            // std::cout << "lost flag " << BoolToString(bbox.lost_flag) << std::endl;

            int object_track_len;
            if (!use_bbox_only_flag)
            {
                object_track_len = obj_obs_ptr->zs.size();
            }
            else
            {
                object_track_len = obj_obs_ptr->zb.size();
            }

            // for debugging
            // std::cout << "object_track_len " << object_track_len << std::endl;

            // check whether the object is lost
            // 1. if bbox tracking is lost
            // 2. if object is being tracked for too long
            if (bbox.lost_flag && object_track_len > min_object_feature_track_length)
            {
                // for debugging
                // std::cout << "lost object id " << bbox.id << std::endl;

                lost_object_ids.push_back(bbox.id);
            }
            else if (object_track_len > max_object_feature_track_length)
            {
                // for debugging
                // std::cout << "max_object_feature_track_length " << max_object_feature_track_length << std::endl;
                // std::cout << "lost object id " << bbox.id << std::endl;
                // std::cout << "lost object class " << bbox.Class << std::endl;

                lost_object_ids.push_back(bbox.id);
            }
            else
            {
                // std::cout << "object track len " << object_track_len << std::endl;
            }

            // insert timestamps into object observations
            obj_obs_ptr->timestamps[cam_id].emplace_back(bbox_kp_msg->header.stamp.toSec());

            // for debugging
            // std::cout << "measurement timestamps " << bbox_kp_msg->header.stamp.toSec() << std::endl;

            Vector4d zb_per_frame;
            zb_per_frame << bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax;
            if (!(zb_per_frame.array().head<2>() < zb_per_frame.array().tail<2>()).all())
            {
                std::stringstream ss;
                ss << "Bad bounding box (maxs are smaller than mins): (xmin, ymin, xmax, ymax): " << zb_per_frame;
                if (zb_per_frame(0) > zb_per_frame(2))
                {
                    std::cout << "[WARN]:" << ss.str() << ". Swamming xmin xmax";
                    std::swap(zb_per_frame(0), zb_per_frame(2));
                }
                else
                {
                    std::swap(zb_per_frame(1), zb_per_frame(3));
                }
                // throw std::runtime_error(ss.str());
            }

            Eigen::Matrix<double, 3, 3> camera_intrinsics;
            cv2eigen(camK, camera_intrinsics);

            assert((zb_per_frame.array().head<2>() < zb_per_frame.array().tail<2>()).all());
            obj_obs_ptr->zb.push_back(normalize_bbox(zb_per_frame, camera_intrinsics));

            if (!use_bbox_only_flag)
            {
                // handle semantic keypoints
                Eigen::MatrixX2d zs_per_frame = Eigen::MatrixXd::Constant(obj_obs_ptr->sem_kp_num, 2, NAN);
                std::vector<int> detected_kp_ids;

                // process detected keypoints
                for (const auto &semkp : bbox_with_kp.keypoints)
                {
                    // object id starts at 1
                    // part label
                    // std::cout << "kp part id " << semkp.semantic_part_label << std::endl;

                    // for debugging
                    // std::cout << "original kp x " << semkp.x + bbox.xmin << std::endl;
                    // std::cout << "original kp y " << semkp.y + bbox.ymin << std::endl;

                    //*********************************************
                    // start keypoint tracking
                    //*********************************************
                    // prevent duplicated kp ids
                    if (std::find(detected_kp_ids.begin(), detected_kp_ids.end(), semkp.semantic_part_label) != detected_kp_ids.end())
                    {
                        /* this kp is detected */
                        // do nothing
                        continue;
                    }

                    detected_kp_ids.push_back(semkp.semantic_part_label);

                    if (!use_unity_dataset_flag)
                    {

                        // for debugging
                        // std::cout << "bbox.xmin " << bbox.xmin << std::endl;
                        // std::cout << "bbox.ymin " << bbox.ymin << std::endl;

                        // for kitti
                        // convert bounding box coordinates to image coordinates
                        obj_obs_ptr->track_sem_kp(semkp.semantic_part_label, semkp.x + bbox.xmin, semkp.y + bbox.ymin, bbox_kp_msg->header.stamp.toSec());
                    }
                    else
                    {
                        // for unity
                        obj_obs_ptr->track_sem_kp(semkp.semantic_part_label, semkp.x, semkp.y, bbox_kp_msg->header.stamp.toSec());
                    }

                    // Convert to opencv format
                    cv::Mat mat(1, 2, CV_32F);

                    // use measurements directly
                    // mat.at<float>(0, 0) = semkp.x;
                    // mat.at<float>(0, 1) = semkp.y;
                    // use tracked kps
                    Eigen::Vector2f pos = obj_obs_ptr->obtain_kp_coord(semkp.semantic_part_label);
                    mat.at<float>(0, 0) = pos(0);
                    mat.at<float>(0, 1) = pos(1);

                    mat = mat.reshape(2); // Nx1, 2-channel

                    // for debugging
                    // std::cout << "before undistort " << mat << std::endl;

                    // only keep measurements inside the image
                    // if (0 > semkp.x || semkp.x > 640)
                    // {
                    //     if (0 > semkp.y || semkp.y > 480)
                    //     {
                    //         continue;
                    //     }
                    // }

                    // Undistort it!
                    cv::undistortPoints(mat, mat, camK, camD);
                    // Construct our return vector
                    Eigen::Vector2d uv_dist;
                    mat = mat.reshape(1); // Nx2, 1-channel
                    uv_dist(0) = mat.at<float>(0, 0);
                    uv_dist(1) = mat.at<float>(0, 1);

                    // for debugging
                    // std::cout << "after undistort " << uv_dist << std::endl;

                    zs_per_frame.row(semkp.semantic_part_label) = uv_dist;
                }

                // process tracked but NOT detected keypoints
                for (auto &tracker : obj_obs_ptr->kp_trackers)
                {
                    if (std::find(detected_kp_ids.begin(), detected_kp_ids.end(), tracker.first) != detected_kp_ids.end())
                    {
                        /* this kp is detected */
                        // do nothing
                    }
                    else
                    {
                        /* this kp is NOT detected */
                        // do prediction only
                        obj_obs_ptr->track_sem_kp(tracker.first, 0, 0, bbox_kp_msg->header.stamp.toSec());
                    }

                    // for debugging
                    // std::cout << "kp id pred only " << tracker.first << std::endl;

                    // Convert to opencv format
                    cv::Mat mat(1, 2, CV_32F);
                    Eigen::Vector2f pos = obj_obs_ptr->obtain_kp_coord(tracker.first);

                    // check if this is a re-detected object
                    if (pos(0) + pos(1) == 0)
                        continue;

                    mat.at<float>(0, 0) = pos(0);
                    mat.at<float>(0, 1) = pos(1);
                    mat = mat.reshape(2); // Nx1, 2-channel

                    // for debugging
                    // std::cout << "before undistort pred only " << mat << std::endl;

                    // Undistort it!
                    cv::undistortPoints(mat, mat, camK, camD);
                    // Construct our return vector
                    Eigen::Vector2d uv_dist;
                    mat = mat.reshape(1); // Nx2, 1-channel
                    uv_dist(0) = mat.at<float>(0, 0);
                    uv_dist(1) = mat.at<float>(0, 1);

                    // for debugging
                    // std::cout << "after undistort " << uv_dist << std::endl;

                    zs_per_frame.row(tracker.first) = uv_dist;
                }

                obj_obs_ptr->zs.push_back(zs_per_frame);
            }

#ifdef SYNC_WITH_IMG 
            if (show_track_image_flag)
            {
                if (!use_bbox_only_flag)
                    obj_obs_ptr->draw_kp_track(track_image);
                
                // always draw bounding box 
                draw_bbox(track_image, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax);
            }
#endif 

        }

#ifdef SYNC_WITH_IMG 
        if (show_track_image_flag)
        {
            // publish track image
            publish_track_image(track_image);
        }
#endif 

    }

    void ObjectInitNode::draw_bbox(cv::Mat &img, double xmin, double ymin, double xmax, double ymax)
    {
        // const float line_thickness = 3;
        const float line_thickness = 4;

        cv::rectangle(img, cv::Point(xmin, ymin),
                      cv::Point(xmax, ymax), cv::Scalar(0, 255, 0), line_thickness);
    }

#ifdef SYNC_WITH_IMG
    void ObjectInitNode::publish_track_image(const cv::Mat &track_image)
    {
        if (trackImagePublisher.getNumSubscribers() < 1)
            return;
 
        cv_bridge::CvImage cvImage;
        cvImage.header.stamp = ros::Time::now();
        cvImage.header.frame_id = child_frame_id;
        cvImage.header.frame_id = "kp_track_image";
        cvImage.encoding = sensor_msgs::image_encodings::BGR8;
        cvImage.image = track_image;

        trackImagePublisher.publish(*cvImage.toImageMsg());

        // for debugging 
        // ROS_INFO_STREAM("Track image has been published.");
    }
#endif

    void ObjectInitNode::visualize_map_only()
    {

        // plot objects
        if (load_gt_object_info_flag)
        {
            publish_gt_objects();
        }

        publish_quadrics();
    }

    void ObjectInitNode::publish_gt_objects()
    {
        visualization_msgs::MarkerArray markers;

        orcvio::vector_eigen<Eigen::MatrixXd> object_position_gt_normalized_vec;
        orcvio::vector_eigen<Eigen::MatrixXd> object_rotation_gt_normalized_vec;

        const int nobjects = all_object_states_dict.size();

        for (unsigned int i = 0; i < object_position_gt_vec.size(); i++)
        {

            visualization_msgs::Marker marker_bbox;
            marker_bbox.id = nobjects + i;
            marker_bbox.header.frame_id = fixed_frame_id;
            marker_bbox.header.stamp = ros::Time::now();
            marker_bbox.type = visualization_msgs::Marker::CUBE;
            marker_bbox.action = visualization_msgs::Marker::ADD;
            marker_bbox.lifetime = ros::Duration();

            // fixed color for all object categories
            marker_bbox.color.r = 0;
            marker_bbox.color.g = 0;
            marker_bbox.color.b = 1;
            // set transparency
            marker_bbox.color.a = 0.3;

            std::string object_class = object_class_gt_vec.at(i);
            if (object_sizes_gt_dict.find(object_class) != object_sizes_gt_dict.end())
            {
                marker_bbox.scale.x = object_sizes_gt_dict[object_class].at(0);
                marker_bbox.scale.y = object_sizes_gt_dict[object_class].at(1);
                marker_bbox.scale.z = object_sizes_gt_dict[object_class].at(2);
            }
            else
            {
                std::cout << "unkown object class " << object_class_gt_vec.at(i) << std::endl;
            }

            Eigen::Vector3d object_position_gt;
            if (use_unity_dataset_flag)
            {
                object_position_gt = object_position_gt_vec.at(i) - first_uav_translation_gt;
            }
            else 
                object_position_gt = object_position_gt_vec.at(i);       

            Matrix3d object_rotation_gt;
            object_rotation_gt = object_rotation_gt_vec.at(i);
            
            object_position_gt_normalized_vec.push_back(object_position_gt);
            object_rotation_gt_normalized_vec.push_back(object_rotation_gt);

            // set object pose
            marker_bbox.pose.position.x = object_position_gt(0, 0);
            marker_bbox.pose.position.y = object_position_gt(1, 0);
            marker_bbox.pose.position.z = object_position_gt(2, 0);

            Eigen::Quaterniond object_q = Eigen::Quaterniond(object_rotation_gt);
            // normalize the quaternion
            object_q = object_q.normalized();
            marker_bbox.pose.orientation.x = object_q.x();
            marker_bbox.pose.orientation.y = object_q.y();
            marker_bbox.pose.orientation.z = object_q.z();
            marker_bbox.pose.orientation.w = object_q.w();

            markers.markers.push_back(marker_bbox);
        }

        if (!gt_object_map_saved_flag)
        {
            save_gt_object_states_to_file(object_id_gt_vec, object_class_gt_vec, object_position_gt_normalized_vec, object_rotation_gt_normalized_vec, result_dir_path_object_map + "gt_object_states.txt");
            gt_object_map_saved_flag = true;
        }

        // Publish
        pub_gt_objects.publish(markers);
    }

    void ObjectInitNode::publish_quadrics()
    {
        // for plotting the ellipsoids
        visualization_msgs::MarkerArray markers;

        for (const auto &object : all_object_states_dict)
        {
            // std::cout << "object id " << object.first << std::endl;

            // for ellipsoid
            visualization_msgs::Marker marker;
            marker.header.frame_id = fixed_frame_id;
            marker.header.stamp = ros::Time::now();
            marker.id = object.second.object_id;
            marker.type = visualization_msgs::Marker::SPHERE;
            marker.action = visualization_msgs::Marker::ADD;
            marker.lifetime = ros::Duration();

            // for keypoints
            visualization_msgs::Marker sphere_list;
            sphere_list.header.frame_id = fixed_frame_id;
            sphere_list.header.stamp = ros::Time::now();
            sphere_list.ns = "spheres";
            sphere_list.action = visualization_msgs::Marker::ADD;
            sphere_list.pose.orientation.w = 1.0;
            sphere_list.id = object.second.object_id;
            sphere_list.type = visualization_msgs::Marker::SPHERE_LIST;

            // set color
            // random color
            // color_from_id(marker.id, marker.color);

            // fixed color
            marker.color.r = object_marker_colors_[object.second.object_class][0];
            marker.color.g = object_marker_colors_[object.second.object_class][1];
            marker.color.b = object_marker_colors_[object.second.object_class][2];

            // color for semantic keypoints is always green
            // for all object classes
            sphere_list.color.r = 0.0f;
            sphere_list.color.g = 1.0f;
            sphere_list.color.b = 0.0f;

            // set transparency
            marker.color.a = 1;
            sphere_list.color.a = 1.0;

            // set shape

            // For the basic shapes, a scale of 1 in all directions means 1 meter on a side
            // ref http://wiki.ros.org/rviz/Tutorials/Markers%3A%20Basic%20Shapes
            marker.scale.x = object.second.ellipsoid_shape(0, 0);
            marker.scale.y = object.second.ellipsoid_shape(1, 0);
            marker.scale.z = object.second.ellipsoid_shape(2, 0);

            // POINTS markers use x and y scale for width/height respectively
            const double point_marker_size = 0.2; 
            // this is big 
            // const double point_marker_size = 0.5; 
            sphere_list.scale.x = point_marker_size;
            sphere_list.scale.y = point_marker_size;
            sphere_list.scale.z = point_marker_size;

            // set pose
            // use the full 6dof estimated pose
            Eigen::Matrix<double, 4, 4> object_pose = object.second.object_pose;

            marker.pose.position.x = object_pose(0, 3);
            marker.pose.position.y = object_pose(1, 3);
            marker.pose.position.z = object_pose(2, 3);

            // convert to quaternion
            Matrix3d R;
            R = object_pose.block(0, 0, 3, 3);
            Eigen::Quaterniond q = Eigen::Quaterniond(R);
            // normalize the quaternion
            q = q.normalized();
            marker.pose.orientation.x = q.x();
            marker.pose.orientation.y = q.y();
            marker.pose.orientation.z = q.z();
            marker.pose.orientation.w = q.w();

            markers.markers.push_back(marker);

            // for plotting the keypoints
            // std::cout << "kp num is " << object.second.object_keypoints_shape_global_frame.rows() << std::endl;
            for (int i = 0; i < object.second.object_keypoints_shape_global_frame.rows(); ++i)
            {
                geometry_msgs::Point p;
                p.x = object.second.object_keypoints_shape_global_frame(i, 0);
                p.y = object.second.object_keypoints_shape_global_frame(i, 1);
                p.z = object.second.object_keypoints_shape_global_frame(i, 2);

                // cannot plot the points that are too large
                if (abs(p.x) < 1e3)
                    sphere_list.points.push_back(p);
                // else
                //     std::cout << "px " << p.x << std::endl;
            }

            // do not plot keypoints if we are only using bbox
            if (!use_bbox_only_flag)
                markers.markers.push_back(sphere_list);

            // text marker
            visualization_msgs::Marker text_marker(marker);
            text_marker.id = 10 * all_object_states_dict.size() + object.second.object_id;
            text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
            text_marker.pose.position.z = marker.pose.position.z + 0.6 * marker.scale.z;
            text_marker.scale.x = 0;
            text_marker.scale.y = 0;
            text_marker.scale.z = 0.5 * marker.scale.z;
            text_marker.color.a = 1;
            text_marker.text = "Id: " + std::to_string(object.second.object_id);
            markers.markers.push_back(text_marker);
        }

        // Publish
        pub_quadrics.publish(markers);
    }

    void ObjectInitNode::save_gt_object_states_to_file(const std::vector<int> &object_id_gt_vec, const std::vector<std::string> &object_class_gt_vec,
                                                       const orcvio::vector_eigen<Eigen::MatrixXd> &object_position_gt_normalized_vec, const orcvio::vector_eigen<Eigen::MatrixXd> &object_rotation_gt_normalized_vec, std::string filepath_format)
    {
        boost::format boost_filepath_format(filepath_format);
        std::ofstream file((boost_filepath_format).str());

        // std::cout << "debug file " << file.is_open() << std::endl;

        if (file.is_open())
        {
            for (unsigned int i = 0; i < object_position_gt_normalized_vec.size(); i++)
            {
                // file << "est object id:\n" << i << '\n';
                file << "gt object id:\n"
                     << object_id_gt_vec.at(i) << '\n';

                std::string object_class = object_class_gt_vec.at(i);
                file << "gt object class:\n"
                     << object_class << '\n';
                file << "gt object sizes:\n"
                     << object_sizes_gt_dict[object_class].at(0) << " "
                     << object_sizes_gt_dict[object_class].at(1) << " " << object_sizes_gt_dict[object_class].at(2) << '\n';

                file << "gt wPq:\n"
                     << object_position_gt_normalized_vec.at(i) << '\n';
                file << "gt wRq:\n"
                     << object_rotation_gt_normalized_vec.at(i) << '\n';
            }
        }
        // else
        //     std::cout << "cannot open file" << std::endl;
    }

    void ObjectInitNode::save_kps_to_file(const int &object_id, const Eigen::Matrix3Xd &valid_shape_global_frame, std::string filepath_format)
    {
        boost::format boost_filepath_format(filepath_format);
        std::ofstream file((boost_filepath_format % object_id).str());

        // std::cout << "debug file " << file.is_open() << std::endl;

        if (file.is_open())
        {
            file << "object id:\n"
                 << object_id << '\n';
            file << "valid_shape_global_frame:\n"
                 << valid_shape_global_frame.transpose() << '\n';
        }
        // else
        //     std::cout << "cannot open file" << std::endl;
    }

    void ObjectInitNode::do_object_feature_initialization(const std::vector<int> &lost_object_ids)
    {

        std::shared_ptr<ObjectFeature> obj_obs_ptr;

        for (const auto &object_id : lost_object_ids)
        {
            obj_obs_ptr = object_obs_dict.at(object_id);

            // find common timestamps in pose and measurements
            std::vector<double> common_clonetimes;
            obtain_common_timestamps(pose_timestamps, obj_obs_ptr->timestamps[0], common_clonetimes);

            // skip if we don't have enough observations
            if (common_clonetimes.size() < min_object_feature_track_length)
            {
                std::cout << "[Object Init Node] not enough observations" << std::endl;
                std::cout << "[Object Init Node] min_object_feature_track_length " << min_object_feature_track_length << std::endl;
                std::cout << "[Object Init Node] common_clonetimes " << common_clonetimes.size() << std::endl;

                continue; 
            }

            if (!use_bbox_only_flag)
            {
                // only keep observations whose timestamp is also in pose
                obj_obs_ptr->clean_old_measurements(common_clonetimes);
            }
            else
            {
                // only keep bounding box observations whose timestamp is also in pose
                obj_obs_ptr->clean_old_measurements_lite(common_clonetimes);
            }

            // Get all timestamps our clones are at (and thus valid measurement times)
            // Create vector of cloned *CAMERA* poses at each of our clone timesteps
            std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> clones_cam;
            // use gt cam poses
            get_times_cam_poses(common_clonetimes, clones_cam);

            // skip if camera motion is small 
            // this is mainly for removing objects 
            // when quadrotor is static in the unity dataset 
            if (use_unity_dataset_flag)
            {
                double motion_thresh = 2; 
                double max_dist = -1; 
                Eigen::Matrix<double,3,1> p_AinG0; 
                for (const auto &pose : clones_cam.at(0))
                {
                    if (max_dist == -1)
                    {
                        p_AinG0 = pose.second.pos_CinG();
                    }
                    double dist = (p_AinG0 - pose.second.pos_CinG()).norm();
                    if (max_dist < dist)
                        max_dist = dist; 
                }
                if (max_dist < motion_thresh)
                {
                    std::cout << "[Object Init Node] not enough motion" << std::endl;
                    std::cout << "[Object Init Node] max motion " << max_dist << std::endl;
                    std::cout << "[Object Init Node] motion thresh " << motion_thresh << std::endl;

                    continue; 
                }
            }

            bool init_success_flag;
            Eigen::Matrix4d wTq;
            std::shared_ptr<ObjectFeatureInitializer> object_feat_init;

            // choose the appropriate object initializer based on object class
            // we only add valid object class so no need to check whether the object class
            // is valid again here
            auto object_feat_init_iter = all_objects_feat_init_.find(obj_obs_ptr->object_class);
            object_feat_init = object_feat_init_iter->second;

            if (!use_bbox_only_flag)
            {
                std::tie(init_success_flag, wTq) = object_feat_init->single_object_initialization(obj_obs_ptr, clones_cam);
            }
            else
            {
                // only use bounding boxes and ignore keypoints
                std::tie(init_success_flag, wTq) = object_feat_init->single_object_initialization_lite(obj_obs_ptr, clones_cam);
            }

            // store the object if initialization is successful
            if (!init_success_flag)
            {
                std::cout << "[Object Init Node] not enough points for Kabsch" << std::endl;
                continue; 
            }

            std::cout << "[Object Init Node] object init success, id: " << object_id << std::endl;

            ObjectState object_state;
            object_state.object_id = object_id;
            object_state.object_class = obj_obs_ptr->object_class;
            object_state.object_pose = wTq;
            object_state.ellipsoid_shape = object_feat_init->getObjectMeanShape();
            object_state.object_keypoints_shape_global_frame = transform_mean_keypoints_to_global(object_feat_init->getObjectKeypointsMeanShape(), wTq);

            all_object_states_dict.insert({object_id, object_state});
            save_object_state_to_file(object_state, common_clonetimes, result_dir_path_object_map + "initial_state_%d.txt");

            // save initial keypoints if not using lite version
            if (!use_bbox_only_flag)
                save_kps_to_file(object_id, object_feat_init->valid_shape_global_frame, result_dir_path_object_map + "estimated_keypoints_%d.txt");

            if (!do_fine_tune_object_pose_using_lm)
            {
                std::cout << "[Object Init Node] ObjectLM not used" << "\n";
                continue; 
            }
            
            bool success;
            if (!use_bbox_only_flag)
            {
                // full object LM
                success = object_feat_init->single_levenberg_marquardt(*obj_obs_ptr, clones_cam, object_state,
                                                                        use_left_perturbation_flag, use_new_bbox_residual_flag);
            }
            else
            {
                // object LM lite using bounding box only
                success = object_feat_init->single_levenberg_marquardt_lite(*obj_obs_ptr, clones_cam, object_state,
                                                                            use_left_perturbation_flag, use_new_bbox_residual_flag);
            }

            if (!success)
            {
                std::cout << "[Object Init Node] ObjectLM failed" << "\n";
                continue; 
            }
            else if (use_bbox_only_flag)
            {
                // when using bbox only, we do it in a loosely coupled way
                // and do not publish object residual to update camera pose
                std::cout << "[Object Init Node] ObjectLM lite version" << "\n";
            }
            else
            {
                // publish residual, jacobian, timestamps here

                orcvio_ros_msgs::ObjectLM object_lm_msg;

                // for timestamps
                object_lm_msg.object_id = object_id;
                object_lm_msg.timestamps = object_feat_init->object_timestamps;

                // for residuals
                std_msgs::Float64MultiArray fvec_all;
                tf::matrixEigenToMsg(object_feat_init->fvec_all, fvec_all);
                object_lm_msg.residual = fvec_all;

                // for jacobians wrt object
                std_msgs::Float64MultiArray fjac_object_state_all;
                tf::matrixEigenToMsg(object_feat_init->fjac_object_state_all, fjac_object_state_all);
                object_lm_msg.jacobian_wrt_object_state = fjac_object_state_all;

                // for jacobians wrt camera
                std_msgs::Float64MultiArray fjac_sensor_state_all;
                tf::matrixEigenToMsg(object_feat_init->fjac_sensor_state_all, fjac_sensor_state_all);
                object_lm_msg.jacobian_wrt_sensor_state = fjac_sensor_state_all;

                // for valid camera poses
                std_msgs::Float64MultiArray valid_camera_poses_all;
                tf::matrixEigenToMsg(object_feat_init->valid_camera_pose_mat, valid_camera_poses_all);
                object_lm_msg.valid_camera_pose_mat = valid_camera_poses_all;

                // for valid zs number
                object_lm_msg.zs_num_wrt_timestamps = object_feat_init->zs_num_wrt_timestamps;

                object_lm_list_msg.object_lm_msgs.push_back(object_lm_msg);

                std::cout << "[Object Init Node] ObjectLM success" << "\n";
            }
            save_object_state_to_file(object_state, common_clonetimes, result_dir_path_object_map + "after_LM_object_state_%d.txt");
        }
    }

    void ObjectInitNode::remove_used_object_obs(std::vector<int> &lost_object_ids)
    {
        for (const auto &object_id : lost_object_ids)
        {
            object_obs_dict.erase(object_id);
        }
        lost_object_ids.clear();
    }

    void ObjectInitNode::color_from_id(const int id, std_msgs::ColorRGBA &color)
    {
        const int SOME_PRIME_NUMBER = 6553;
        int id_dep_num = ((id * SOME_PRIME_NUMBER) % 255); // id dependent number generation
        color.r = (255.0 - id_dep_num) / 255.0;
        color.g = 0.5;
        color.b = (id_dep_num) / 255.0;
    }

    // void ObjectInitNode::obtain_common_timestamps(const std::vector<double>& pose_timestamps_all, const std::vector<double>& mea_timestamps_all, std::vector<double>& common_clonetimes)
    void ObjectInitNode::obtain_common_timestamps(std::vector<double> &pose_timestamps_all, std::vector<double> &mea_timestamps_all, std::vector<double> &common_clonetimes)
    {

        // for debugging
        // std::cout << "pose timestamps " << std::endl;
        // for (const auto& tp : pose_timestamps_all)
        // {
        //     std::cout << tp << ", ";
        // }
        // std::cout << std::endl;
        // std::cout << "measurement timestamps " << std::endl;
        // for (const auto& tp : mea_timestamps_all)
        // {
        //     std::cout << tp << ", ";
        // }
        // std::cout << std::endl;

        // Sort the vector
        // TODO is this necessary?
        // std::sort(pose_timestamps_all.begin(), pose_timestamps_all.end());
        // std::sort(mea_timestamps_all.begin(), mea_timestamps_all.end());

        std::vector<double> v(pose_timestamps_all.size() + mea_timestamps_all.size());
        std::vector<double>::iterator it, st;
        it = set_intersection(pose_timestamps_all.begin(),
                              pose_timestamps_all.end(),
                              mea_timestamps_all.begin(),
                              mea_timestamps_all.end(),
                              v.begin());

        // for debugging
        // std::cout << "common timestamps " << std::endl;
        // for (st = v.begin(); st != it; ++st)
        // {
        //     std::cout << *st << ", ";
        //     common_clonetimes.push_back(*st);
        // }
        // std::cout << std::endl;

        for (st = v.begin(); st != it; ++st)
        {
            common_clonetimes.push_back(*st);
        }

        // common_clonetimes = pose_timestamps_all;
    }

    void ObjectInitNode::get_times_cam_poses(const std::vector<double> &clonetimes, std::unordered_map<size_t, std::unordered_map<double, FeatureInitializer::ClonePose>> &clones_cam)
    {

        if (clones_imu.size() == 0)
        {
            throw std::runtime_error("No groundtruth poses received, check the groundtruth topic!");
        }

        size_t cam_id = 0;

        // For this camera, create the vector of camera poses
        std::unordered_map<double, FeatureInitializer::ClonePose> clones_cami;

        for (const auto &timestamp : clonetimes)
        {

            // Get the position of this clone in the global
            Eigen::Matrix<double, 3, 3> R_ItoG = clones_imu.at(timestamp).Rot_GtoC();
            Eigen::Matrix<double, 3, 1> p_IinG = clones_imu.at(timestamp).pos_CinG();

            Eigen::Matrix<double, 4, 4> T_ItoG;
            T_ItoG.block(0, 0, 3, 3) = R_ItoG;
            T_ItoG.block(0, 3, 3, 1) = p_IinG;

            Eigen::Matrix<double, 4, 4> T_CitoG;
            T_CitoG = T_ItoG * T_CtoI;

            Eigen::Matrix<double, 3, 3> R_GtoCi;
            R_GtoCi = T_CitoG.block(0, 0, 3, 3).transpose();

            Eigen::Matrix<double, 3, 1> p_CioinG;
            p_CioinG = T_CitoG.block(0, 3, 3, 1);

            // Append to our map
            clones_cami.insert({timestamp, FeatureInitializer::ClonePose(R_GtoCi, p_CioinG)});
        }

        // Append to our map
        clones_cam.insert({cam_id, clones_cami});

        return;
    }

    bool ObjectInitNode::callback_objectLMBlocking(
        orcvio_ros_msgs::ObjectLMResults::Request &req,
        orcvio_ros_msgs::ObjectLMResults::Response &res)
    {

        // optimize objects that are lost
        if (lost_object_ids.size() > 0)
        {
            // for debugging
            // for (const auto & id : lost_object_ids)
            //     std::cout << "lost object id " << id << std::endl;

            //*********************************************
            // wait for this function to finish before any prediction or update in VIO
            //*********************************************
            do_object_feature_initialization(lost_object_ids);

            remove_used_object_obs(lost_object_ids);
        }

        if (object_lm_list_msg.object_lm_msgs.size() != 0)
        {
            std::cout << "[Object Init Node] Publish " << object_lm_list_msg.object_lm_msgs.size() << " object messages"
                      << "\n";
            object_lm_list_msg.object_lm_msgs.clear();
        }

        res.msg = object_lm_list_msg;

        // publish the markers
        visualize_map_only();

        return true;
    }

} // namespace orcvio