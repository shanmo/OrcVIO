#include <memory>
#include <functional>
#include <mutex>
#include <queue>

#include <boost/range/counting_range.hpp>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp> // cv::*

#include <ros/ros.h>
#include <std_msgs/String.h>
#include <nodelet/nodelet.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/subscriber.h>
#include <sort_ros/TrackedBoundingBoxes.h>
#include <cv_bridge/cv_bridge.h>
#include <starmap_ros_msgs/SemanticKeypointWithCovariance.h>
#include <starmap_ros_msgs/TrackedBBoxListWithKeypoints.h>
#include <starmap_ros_msgs/TrackedBBoxWithKeypoints.h>
#include <starmap/starmap.h>
#include <boost/filesystem.hpp>
#include <image_transport/subscriber_filter.h>

using namespace std;
using cv::Mat;
using cv::Vec3f;
namespace bfs = boost::filesystem;
using boost::format;

namespace starmap
{

  template<typename T, typename... Args>
  std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
  }


  class Starmap : public nodelet::Nodelet
  {
  public:
    Starmap() : sub_(10) {}

  private:
  virtual void onInit() {
    NODELET_WARN("Initializing ");
    namespace sph = std::placeholders; // for _1, _2, ...
    std::string image_topic, bbox_topic, keypoint_topic, visualization_topic,
    starmap_model_path;
    int gpu_id;
    auto nh = getNodeHandle();
    auto private_nh = getPrivateNodeHandle();
    NODELET_DEBUG("Got node handles");
    if ( ! private_nh.getParam("starmap_model_path", starmap_model_path) ) {
      NODELET_FATAL("starmap_model_path is required");
      throw std::runtime_error("starmap_model_path is required");
    }
    if ( ! bfs::exists(starmap_model_path) ) {
      NODELET_FATAL("starmap_model_path '%s' does not exists", starmap_model_path.c_str());
      throw std::runtime_error("starmap_model_path does not exists");
    }
    private_nh.param<std::string>("image_topic", image_topic, "image");
    private_nh.param<bool>("to_color_img_flag", to_color_img_flag_, false);
    private_nh.param<std::string>("bbox_topic", bbox_topic, "bounding_boxes");
    private_nh.param<std::string>("keypoint_topic", keypoint_topic, "keypoints");
    private_nh.param<std::string>("visualization_topic", visualization_topic, "visualization");
    private_nh.param<int>("gpu_id", gpu_id, -1);
    private_nh.param("draw_labels", draw_labels_, true);
    timer_ = nh.createTimer(ros::Duration(0.05),
                            std::bind(& Starmap::timerCb, this, sph::_1));

    auto model = starmap::model_load(starmap_model_path, gpu_id);

    NODELET_INFO("Subscribing to %s", image_topic.c_str());
    image_trans_ = make_unique<image_transport::ImageTransport>(private_nh);
    image_sub_.subscribe(nh, image_topic, 10);
    bbox_sub_.subscribe(nh, bbox_topic, 10);
    sub_.connectInput(image_sub_, bbox_sub_);
    sub_.registerCallback(std::bind(&Starmap::messageCb, this, model, sph::_1, sph::_2));
    pub_ = private_nh.advertise<starmap_ros_msgs::TrackedBBoxListWithKeypoints>(keypoint_topic, 10);

    vis_ = image_trans_->advertise(visualization_topic, 10);

  }

  cv::Rect2i safe_rect_bbox(const sort_ros::TrackedBoundingBox& bbox,
                            const cv::Mat& image) {
    int xmin = max<int>(0, bbox.xmin);
    int ymin = max<int>(0, bbox.ymin);
    int xmax = min<int>(image.cols, bbox.xmax);
    int ymax = min<int>(image.rows, bbox.ymax);
    cv::Rect2i bbox_rect(xmin, ymin, xmax - xmin, ymax - ymin);
    return bbox_rect;
  }

  void  visualize_all_bbox(cv::Mat& image,
                           const starmap_ros_msgs::TrackedBBoxListWithKeypointsConstPtr& bbox_with_kp_list)
  {
    // std::vector<SemanticKeypoint> all_semkp_list;
    for (auto& bbox_with_kp : bbox_with_kp_list->bounding_boxes) {
      auto& bbox = bbox_with_kp.bbox;

      auto bbox_rect = safe_rect_bbox(bbox_with_kp.bbox, image);
      if (bbox_rect.area() < 1)
        continue;
      int id_dep_num = ((bbox.id * 6553) % 255); // id dependent number generation
      cv::Scalar color(255 - id_dep_num, id_dep_num, 255 - id_dep_num);
      cv::rectangle(image, bbox_rect, color, 2);
      cv::putText(image, (format("id: %d") % bbox.id).str(),
                  {bbox_rect.x, bbox_rect.y},
                  cv::FONT_HERSHEY_SIMPLEX,
                  std::max(0.4, 2.0 * bbox_rect.height / image.rows),
                  color, 1);
      auto bboxroi = image(bbox_rect);
      Points pts;
      std::vector<string> label_list;
      std::vector<SemanticKeypoint> bbox_semkp_list;
      for (auto& semkp: bbox_with_kp.keypoints) {
        cv::Point2i pti(semkp.x, semkp.y);
        pts.emplace_back(pti);
        label_list.emplace_back(semkp.semantic_part_label_name);

        cv::Point2i gkp = pti; // + 
        SemanticKeypoint skp;
        skp.pos2d = gkp;
        skp.label = semkp.semantic_part_label_name;
        bbox_semkp_list.push_back(skp);
      }
      starmap::visualize_keypoints(bbox.Class, bboxroi, bbox_semkp_list,
                                   /*draw_labels=*/draw_labels_);
      // for (auto const & skp : bbox_semkp_list) {
      //   SemanticKeypoint skp2( skp );
      //   skp2.pos2d = skp.pos2d + cv::Point2i(bbox_rect.x, bbox_rect.y);
      //   all_semkp_list.push_back(skp2);
      // }
    }
    // starmap::visualize_keypoints(image, all_semkp_list, /*draw_labels=*/draw_labels_);
  }


  // must use a ConstPtr callback to use zero-copy transport
  void messageCb(torch::jit::script::Module model,
                 const sensor_msgs::ImageConstPtr& message,
                 const sort_ros::TrackedBoundingBoxesConstPtr& bboxes) {
    NODELET_INFO("Callback called ... ");
    auto private_nh = getPrivateNodeHandle();
    int input_res;
    private_nh.param<int>("input_res", input_res, 256);
    bool visualize;
    private_nh.param<bool>("visualize", visualize, false);

    // add support for grayscale image 
    cv_bridge::CvImageConstPtr img = cv_bridge::toCvShare(message);
    cv::Mat img_mat = img->image;  
    if (to_color_img_flag_)
    {
      // convert to rgb image if necessary
      cv::cvtColor(img_mat, img_mat, cv::COLOR_GRAY2RGB);
    }

    starmap_ros_msgs::TrackedBBoxListWithKeypointsPtr bbox_with_kp_list =
      boost::make_shared<starmap_ros_msgs::TrackedBBoxListWithKeypoints>();
    bbox_with_kp_list->header.stamp = message->header.stamp;
    bbox_with_kp_list->header.frame_id = message->header.frame_id;
    for (auto& bbox: bboxes->bounding_boxes) {
      // auto bbox_rect = safe_rect_bbox(bbox, img->image);
      auto bbox_rect = safe_rect_bbox(bbox, img_mat);
      starmap_ros_msgs::TrackedBBoxWithKeypoints bbox_with_kp;
      if (bbox_rect.area() >= 1) {
        // auto bboxroi = img->image(bbox_rect);
        auto bboxroi = img_mat(bbox_rect);
        Mat bboxfloat;
        bboxroi.convertTo(bboxfloat, CV_32FC3, 1/255.0);
        NODELET_DEBUG("Calling  model ... ");
        vector<SemanticKeypoint> semkp_list =
          find_semantic_keypoints_prob_depth(model, bbox.Class, bboxfloat, input_res,
                                            /*visualize=*/visualize);

        bbox_with_kp.bbox = bbox; // Duplicate information
        for (auto const& semkp: semkp_list) {
          auto& pt = semkp.pos2d;
          starmap_ros_msgs::SemanticKeypointWithCovariance kpt;
          kpt.x = pt.x;
          kpt.y = pt.y;
          kpt.cov.insert(kpt.cov.end(), semkp.cov.val,  semkp.cov.val + semkp.cov.rows * semkp.cov.cols);
          kpt.semantic_part_label_name = semkp.label;
          kpt.semantic_part_label =
            starmap::GLOBAL_OBJECT_STRUCTURE.get_label_index(bbox.Class, semkp.label);
          bbox_with_kp.keypoints.push_back(kpt);
        }
      }
      bbox_with_kp_list->bounding_boxes.push_back(bbox_with_kp);
    }
    pub_.publish(bbox_with_kp_list);
    if (vis_.getNumSubscribers() >= 1) {
      const std::lock_guard<std::mutex> lock(image_to_publish_mutex_);
      if (input_img_bbox_queue_.size() < max_queue_size_) {
        // cv::Mat vis = img->image.clone();
        cv::Mat vis = img_mat.clone();
        input_img_bbox_queue_.emplace(vis, bbox_with_kp_list);
      }
    }
  }

  void timerCb(const ros::TimerEvent &) {
    cv::Mat vis;
    starmap_ros_msgs::TrackedBBoxListWithKeypointsConstPtr bbox_with_kp_list;
    {
      const std::lock_guard<std::mutex> lock(image_to_publish_mutex_);
      if ( ! input_img_bbox_queue_.empty()) {
        std::tie(vis, bbox_with_kp_list) = input_img_bbox_queue_.front();
        input_img_bbox_queue_.pop();
      }
    }

    // for visualization 
    if (bbox_with_kp_list && vis.data != nullptr) {
      visualize_all_bbox(vis, bbox_with_kp_list);
      cv_bridge::CvImage cvImage;
      // cvImage.header.stamp = bbox_with_kp_list->header.stamp;
      cvImage.header.stamp = ros::Time::now();
      cvImage.header.frame_id = bbox_with_kp_list->header.frame_id;
      cvImage.encoding = sensor_msgs::image_encodings::BGR8;
      cvImage.image = vis;

      NODELET_DEBUG("publishing visualization... ");
      vis_.publish(cvImage.toImageMsg());
    }

  }

  message_filters::Subscriber<sensor_msgs::Image> image_sub_;
  message_filters::Subscriber<sort_ros::TrackedBoundingBoxes> bbox_sub_;
  message_filters::TimeSynchronizer<sensor_msgs::Image, sort_ros::TrackedBoundingBoxes> sub_;
  ros::Publisher pub_;
  image_transport::Publisher vis_;
  std::unique_ptr<image_transport::ImageTransport> image_trans_;
  ros::Timer timer_;
  std::queue<std::tuple<
               cv::Mat,
               starmap_ros_msgs::TrackedBBoxListWithKeypointsConstPtr>
             > input_img_bbox_queue_;
  std::mutex image_to_publish_mutex_;
  int max_queue_size_ = 10;
  bool draw_labels_;
  bool to_color_img_flag_;
};

} // namespace Starmap

PLUGINLIB_EXPORT_CLASS( starmap::Starmap, nodelet::Nodelet );

