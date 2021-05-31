#ifndef STARMAP_STARMAP_H
#define STARMAP_STARMAP_H

#include <type_traits>
#include <tuple>

#include "opencv2/opencv.hpp"
#include "torch/script.h"

namespace starmap {

template<typename _Tp, int m, int n>
  cv::Matx<_Tp, m, n> to_cvmatx(std::initializer_list<_Tp> list) {
   assert(list.size() == m*n);
   std::vector<_Tp> values(list);
   cv::Matx<_Tp, m, n> mat(values.data());
   return mat;
}

cv::Mat crop(const cv::Mat& img,
            const int desired_side);

typedef std::vector<cv::Point2i> Points;
Points run_starmap_on_img(const std::string& starmap_filepath,
                          const std::string& img_filepath,
                          const int input_res,
                          const int gpu_id,
                          const bool visualize = true);


struct SemanticKeypoint {
  cv::Matx22f cov_from_heatmap(cv::Mat const & hm_patch);
  SemanticKeypoint(cv::Point2i const & p,
                   cv::Vec3f const & x,
                  const float d,
                   cv::Mat const &  h,
                  std::string const& l)
    : pos2d(p),
    xyz(x),
    depth(d),
    hm_patch(h),
    cov(cov_from_heatmap(h)),
    label(l)
  {}

  SemanticKeypoint(cv::Point2i const & p,
                  cv::Vec3f const & x,
                  const float d,
                  const cv::Mat& h,
                  cv::Matx22f const & cov,
                  std::string const & l)
    : pos2d(p),
      xyz(x),
      depth(d),
      hm_patch(h),
      cov(cov),
      label(l)
    {}

  SemanticKeypoint()
    : pos2d{0, 0},
    xyz{0, 0, 0},
    depth(0),
    hm_patch(to_cvmatx<float, 3, 3>({0, 0, 0, 0, 1, 0, 0, 0, 0})),
    cov(cv::Matx22f::eye()),
    label("")
  {}

  static SemanticKeypoint Zero() {
    SemanticKeypoint z;
    return z;
  }


  SemanticKeypoint operator+ (const SemanticKeypoint& other) const {
    if ( (other.label != "") && (label != "") && label != other.label ) {
      throw std::runtime_error("label must be the same got: " + label + " other: " + other.label);
    }

    cv::Matx22f sumcov{1, 0, 0, 1};
    cv::Matx21f myeigvals, other_eigvals;
    cv::eigen(cov, myeigvals);
    cv::eigen(other.cov, other_eigvals);
    sumcov(0, 0) = std::max(myeigvals(0, 0), other_eigvals(0, 0));
    sumcov(1, 1) = std::max(myeigvals(1, 0), other_eigvals(1, 0));
    SemanticKeypoint sum(pos2d + other.pos2d,
                         xyz + other.xyz,
                         depth + other.depth,
                         (hm_patch + other.hm_patch) / 2.0,
                         sumcov,
                         (label == "") ? other.label : label);
    return sum;
  }

  SemanticKeypoint operator/ (const float div) const {
    SemanticKeypoint q(pos2d / div,
                       xyz / div,
                       depth / div,
                       hm_patch,
                       label);
    return q;
  }

  cv::Point2i pos2d;
  cv::Vec3f xyz;
  float depth;
  cv::Mat hm_patch;
  cv::Matx22f cov;
  std::string label;
};

std::ostream& operator<< (std::ostream& o, const SemanticKeypoint& semkp);


//std::tuple<Points, std::vector<std::string>, std::vector<float>, std::vector<float>>
std::vector<SemanticKeypoint>
 find_semantic_keypoints_prob_depth(torch::jit::script::Module model,
                                    const std::string object_class, 
                                    const cv::Mat& img,
                                    const int input_res,
                                    const bool visualize,
                                    const bool unique_labels = true);

cv::Mat nms(const cv::Mat& det, const int size = 3);

torch::jit::script::Module
  model_load(const std::string& model_path, const int gpu_id);

std::tuple<cv::Mat, cv::Mat, cv::Mat>
  model_forward(torch::jit::script::Module model,
                const cv::Mat& imgfloat);

 std::vector<cv::Point2i> parse_keypoints_from_heatmap(cv::Mat & det, const float thresh = 0.05, const int border_threshold = 1);

 void visualize_keypoints(const std::string& object_class,
                          cv::Mat& vis,
                          const std::vector<SemanticKeypoint>& semkp_list,
                          bool draw_labels = false);


/**
* Represents canonical semantic points of:
  car, 
  chair,
  bicycle,
  monitor,
  table  
*/
class ObjectStructure {
public:
  ObjectStructure();
  const std::string&
    find_semantic_part(const std::string& object_class, const cv::Matx<float, 3, 1>& cam_view_feat) const;
  const cv::Scalar get_label_color(const std::string& object_class, const std::string& label) const;
  const size_t get_label_index(const std::string& object_class, const std::string& label) const;

protected:
  // for car 
  const cv::Matx<float, 12, 3> car_canonical_points_;
  const std::vector<std::string> car_labels_;

  // for chair 
  const cv::Matx<float, 10, 3> chair_canonical_points_;
  const std::vector<std::string> chair_labels_;

  // for bicycle 
  const cv::Matx<float, 11, 3> bicycle_canonical_points_;
  const std::vector<std::string> bicycle_labels_;

  // for monitor
  const cv::Matx<float, 8, 3> monitor_canonical_points_;
  const std::vector<std::string> monitor_labels_;

  // for table 
  const cv::Matx<float, 12, 3> table_canonical_points_;
  const std::vector<std::string> table_labels_;

  // for all objects 
  // max number of color is 12 
  const cv::Matx<uint8_t, 12, 3> all_colors_;
};

static const ObjectStructure GLOBAL_OBJECT_STRUCTURE;

}

#endif // STARMAP_STARMAP_H
