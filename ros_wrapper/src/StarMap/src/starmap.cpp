#include <initializer_list>
#include <type_traits> // enable_if
#include <tuple> // tuple, tie
#include <cmath> // floor
#include <algorithm> // max
#include <unordered_map>
#include <boost/range/counting_range.hpp>
#include <boost/format.hpp>
#include <limits>

#include "starmap/starmap.h" // starmap
#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/core/mat.hpp"
#include "torch/script.h"

void gsl_Ensures(bool condition) {
  assert(condition);
}

void gsl_Expects(bool condition) {
  assert(condition);
}


using std::vector;
using std::string;
using std::unordered_map;
using std::cout;
using std::tuple;
using std::tie;
using std::max;
using std::make_tuple;
using std::swap;
using cv::Mat;
using cv::Matx;
using cv::Scalar;
using cv::circle;
using cv::imread;
using cv::imwrite;
using cv::imshow;
using cv::waitKey;
using cv::IMREAD_COLOR;
using cv::findNonZero;
using cv::Point2i;
using cv::Point2f;
using cv::Point3f;
using cv::Vec3f;
using cv::Rect;
using cv::Rect2f;
using cv::Range;
using cv::Size;
using cv::cvtColor;
using cv::COLOR_GRAY2BGR;
using boost::format;

namespace starmap {

ObjectStructure::ObjectStructure() :

    // for car 
    car_canonical_points_ ( to_cvmatx<float, 12, 3>({
                                                -0.09472257, -0.07266671,  0.10419698,
                                                0.09396329, -0.07186594,  0.10468729,
                                                0.100639  , 0.26993483, 0.11144333,
                                                -0.100402 ,  0.2699945,  0.111474 ,
                                                -0.12014713, -0.40062513, -0.02047777,
                                                0.1201513 , -0.4005558 , -0.02116918,
                                                0.12190333, 0.40059162, 0.02385612,
                                                -0.12194733,  0.40059462,  0.02387712,
                                                -0.16116614, -0.2717491 , -0.07981283,
                                                -0.16382502,  0.25057048, -0.07948726,
                                                0.1615844 , -0.27168764, -0.07989835,
                                                0.16347528,  0.2507412 , -0.07981754 })),
    car_labels_{
            "upper_left_windshield",
            "upper_right_windshield",
            "upper_right_rearwindow",
            "upper_left_rearwindow",
            "left_front_light",
            "right_front_light",
            "right_back_trunk",
            "left_back_trunk",
            "left_front_wheel",
            "left_back_wheel",
            "right_front_wheel",
            "right_back_wheel"},

    // for chair 
    chair_canonical_points_ ( to_cvmatx<float, 10, 3>({
                      -0.15692863, 0.16241859, -0.28825333,
                      -0.1798645 , 0.12591167, -0.005276,
                      0.1585531 , 0.16203016, -0.28810933,
                      0.1746565 , 0.12572383, -0.005172,
                      -0.16835921, -0.18672756, -0.28799594,
                      -0.18498084, -0.20056212, -0.00706541,
                      0.17062087, -0.18674056, -0.28799594,
                      0.18131248, -0.19883625, -0.00708729,
                      -0.19126813, 0.18548547, 0.3550319,
                      0.18972683, 0.1843335 , 0.355234
                       })),
    chair_labels_{
            "leg_upper_left",
            "seat_upper_left",
            "leg_upper_right",
            "seat_upper_right",
            "leg_lower_left",
            "seat_lower_left",
            "leg_lower_right",
            "seat_lower_right",
            "back_upper_left",
            "back_upper_right"
            },

    // for bicycle  
    bicycle_canonical_points_ ( to_cvmatx<float, 11, 3>({
                      -1.32916667e-04, -1.60928750e-01, 2.01763333e-01,
                      -0.11672605, -0.16468605, 0.19614239,
                      0.11672303, -0.16524933, 0.19602163,
                      -0.02192129, -0.26586988, -0.08524955,
                      0.02273579, -0.26586488, -0.08522026,
                      -1.34333333e-04, 2.03842000e-01, 1.72801000e-01,
                      -1.20833333e-05, 3.85917857e-02, 1.77299643e-01,
                      -0.02799391, 0.26613146, -0.08263057,
                      0.02527268, 0.26451735, -0.08285319,
                      -0.03516337, 0.02347924, -0.09574672,
                      0.03071458, 0.0243553 , -0.094824
                       })),
    bicycle_labels_{
            "head_center",
            "left_handle",
            "right_handle",
            "left_front_wheel",
            "right_front_wheel",
            "seat_back",
            "seat_front",
            "left_back_wheel",
            "right_back_wheel",
            "left_pedal_center",
            "right_pedal_center"
            },

    // for monitor 
    monitor_canonical_points_ ( to_cvmatx<float, 8, 3>({
                      -0.3585725 , -0.14129875, -0.21807875,
                      0.359125 , -0.14131625, -0.21808875,
                      -0.358585 , -0.1412875, 0.276735,
                      0.35913625, -0.1413175 , 0.27672,
                      -0.1128375, 0.125925 , -0.0737725,
                      0.1128375, 0.125925 , -0.0737725,
                      -0.1128375, 0.125925 , 0.075065,
                      0.1128375, 0.125925 , 0.075065
                       })),
    monitor_labels_{
            "front_bottom_left",
            "front_bottom_right",
            "front_top_left",
            "front_top_right",
            "back_bottom_left",
            "back_bottom_right",
            "back_top_left",
            "back_top_right"
            },

    // for table 
    table_canonical_points_ ( to_cvmatx<float, 12, 3>({
                      -0.30718615, 0.19259487, 0.13099417,
                      0.30625051, 0.19245705, 0.13087167,
                      0.30588192, -0.19297513, 0.13099417,
                      -0.30671038, -0.19352385, 0.13099417,
                      -0.28230901, 0.15100807, -0.1288375,
                      0.28118408, 0.14851215, -0.1284475,
                      0.28159246, -0.16839029, -0.12771,
                      -0.28178158, -0.16882001, -0.12771,
                      0. , 0.05205833, 0.03904333,
                      -0.05205833, 0. , 0.03904333,
                      0. , -0.05205833, 0.03904333,
                      0.05205833, 0. , 0.03904333
                       })),
    table_labels_{
            "top_upper_left",
            "top_upper_right",
            "top_lower_right",
            "top_lower_left",
            "leg_upper_left",
            "leg_upper_right",
            "leg_lower_right",
            "leg_lower_left",
            "top_up",
            "top_left",
            "top_down",
            "top_right"
            },

    // for all objects 
    /// Colors in BGR for corresponding labels
    all_colors_ ( to_cvmatx<uint8_t, 12, 3>({ 0, 0, 0,
                                          0, 0, 128,
                                          0, 0, 255,
                                          0, 128, 0,
                                          0, 128, 128,
                                          0, 128, 255,
                                          0, 255, 0,
                                          0, 255, 128,
                                          0, 255, 255,
                                          255, 0, 0,
                                          255, 0, 128,
                                          255, 0, 255 }))
  {
  }


const std::string&
  ObjectStructure::find_semantic_part(const std::string& object_class, const cv::Matx<float, 3, 1>& cam_view_feat) const
{

  size_t min_index;
  float min_dist = std::numeric_limits<float>::max();
  float dist; 
  cv::Mat canonical_points;
  
  if (object_class == "car" || object_class == "truck" || object_class == "bus")
  {
    // copy matx values to mat 
    canonical_points = cv::Mat(car_canonical_points_, true);
  }
  else if (object_class == "chair" || object_class == "bench" || object_class == "sofa" )
  {
    // copy matx values to mat 
    canonical_points = cv::Mat(chair_canonical_points_, true);
  }
  else if (object_class == "bicycle")
  {
    // copy matx values to mat 
    canonical_points = cv::Mat(bicycle_canonical_points_, true);
  }
  else if (object_class == "tvmonitor" || object_class == "laptop" || object_class == "computer")
  {
    // copy matx values to mat 
    canonical_points = cv::Mat(monitor_canonical_points_, true);
  }
  else if (object_class == "diningtable")
  {
    // copy matx values to mat 
    canonical_points = cv::Mat(table_canonical_points_, true);
  }

  Matx<float, 1, 3> cam_view_feat_mat(cam_view_feat.reshape<1, 3>());
  for (size_t i = 0; i < canonical_points.rows; ++i) {
    dist = cv::norm((cam_view_feat_mat - canonical_points.row(i)), cv::NORM_L2SQR);
    if (min_dist > dist)
    {
      min_dist = dist; 
      min_index = i; 
    }
  }

  if (object_class == "car" || object_class == "truck" || object_class == "bus")
  {
    return car_labels_[min_index];
  }
  else if (object_class == "chair" || object_class == "bench" || object_class == "sofa" )
  {
    return chair_labels_[min_index];
  }
  else if (object_class == "bicycle")
  {
    return bicycle_labels_[min_index];
  }
  else if (object_class == "tvmonitor" || object_class == "laptop" || object_class == "computer")
  {
    return monitor_labels_[min_index];
  }  
  else if (object_class == "diningtable")
  {
    return table_labels_[min_index];
  } else {
      throw std::runtime_error("Unknown object_class");
  }
}

const cv::Scalar
  ObjectStructure::get_label_color(const std::string& object_class, const std::string& label) const
{
  auto col = all_colors_.row(get_label_index(object_class, label));
  return cv::Scalar(col(0,0), col(0,1), col(0,2));
}


const size_t
  ObjectStructure::get_label_index(const std::string& object_class, const std::string& label) const
{

  size_t label_ind;
  if (object_class == "car" || object_class == "truck" || object_class == "bus")
  {
    auto it = std::find(car_labels_.begin(), car_labels_.end(), label);
    label_ind = std::distance(car_labels_.begin(), it);
  }
  else if (object_class == "chair" || object_class == "bench" || object_class == "sofa" )
  {
    auto it = std::find(chair_labels_.begin(), chair_labels_.end(), label);
    label_ind = std::distance(chair_labels_.begin(), it);
  }
  else if (object_class == "bicycle")
  {
    auto it = std::find(bicycle_labels_.begin(), bicycle_labels_.end(), label);
    label_ind = std::distance(bicycle_labels_.begin(), it);
  }
  else if (object_class == "tvmonitor" || object_class == "laptop" || object_class == "computer")
  {
    auto it = std::find(monitor_labels_.begin(), monitor_labels_.end(), label);
    label_ind = std::distance(monitor_labels_.begin(), it);
  }
  else if (object_class == "diningtable")
  {
    auto it = std::find(table_labels_.begin(), table_labels_.end(), label);
    label_ind = std::distance(table_labels_.begin(), it);
  }

  return label_ind;

}


double scale_for_crop(const Point2i& img_size,
                      const int desired_side)
{
  int max_side = max(img_size.x, img_size.y);
  double scale_factor = static_cast<double>(desired_side) / static_cast<double>(max_side);
  gsl_Ensures(scale_factor > 0);
  return scale_factor;
}


Points convert_to_precrop(const Points& keypoints,
                          const Point2i& pre_crop_size,
                          const int desired_side,
                          const double addnl_scale_factor )
{
  Points pre_crop_kp_vec;
  Point2i curr_size(desired_side, desired_side);
  double scale_factor = scale_for_crop(pre_crop_size, desired_side);
  for (auto& kp: keypoints) {
    Point2i pre_crop_kp = (kp * addnl_scale_factor - curr_size / 2) / scale_factor
      + pre_crop_size / 2;
    pre_crop_kp_vec.push_back(pre_crop_kp);
  }
  return pre_crop_kp_vec;
}


/**
 * @brief crop img to a square image with each side as desired side
 *
 * @param img            The image to crop
 * @param desired_side   Desired size of output cropped image
 * @return Converted image
 */
Mat crop(const Mat& img,
             const int desired_side)
{
    constexpr int D = 2;
    gsl_Expects(desired_side > 0);
    gsl_Expects(img.dims >= 2 && img.dims <= 3);
    double scale_factor = scale_for_crop({img.size[1], img.size[0]}, desired_side);
    // scale the image first
    Mat resized_img;
    resize(img, resized_img,
               Size((img.size[1] * scale_factor),
                    (img.size[0] * scale_factor)));

    // Cropping begins here
    // The image rectangle clockwise
    Rect2f rect_resized(0, 0, resized_img.size[1], resized_img.size[0]);
    auto resized_max_side = max(resized_img.size[0], resized_img.size[1]);


    // Project the rectangle from source image to target image
    // TODO account for rotation
    Point2f target_center( desired_side / 2, desired_side / 2);
    Point2f resized_img_center( resized_img.size[1] / 2, resized_img.size[0] / 2);
    auto translate = target_center - resized_img_center ;
    auto rect_target = (rect_resized + translate);

    // img.size[2] might not be accessible
    const int size[3] = {desired_side, desired_side, img.size[2]};
    Mat output_img = Mat::zeros(img.dims, size, img.type());
    Mat output_roi(output_img, rect_target);
    Mat source_roi = resized_img(Rect(0, 0, rect_target.width, rect_target.height));
    source_roi.copyTo(output_roi);
    return output_img;
}

/**
 * @brief nms
 * @param det
 * @param size
 * @return
 */
Mat nms(const Mat& det, const int size) {
  gsl_Expects(det.type() == CV_32F);
  Mat pooled = Mat::zeros(det.size(), det.type());
  int start = size / 2;
  for (int i = start; i < det.size[0] - start; ++i) {
    for (int j = start; j < det.size[1] - start; ++j) {
      Mat window = det(Range(i-start, i-start+size),
                       Range(j-start, j-start+size));
      double minval, maxval;
      minMaxLoc(window, &minval, &maxval);
      // auto mele = max_element(window.begin<float>(), window.end<float>());
      pooled.at<float>(i, j) = maxval;
    }
  }
  // Suppress the non-max parts
  Mat nonmax = pooled != det;
  pooled.setTo(0, nonmax);
  return pooled;
}

/**
 * @brief Parse heatmap for points above a threshold
 *
 * @param det     The heatmap to parse
 * @param thresh  Threshold over which points are kept
 * @return        Vector of points above threshold
 */
vector<Point2i>
parse_keypoints_from_heatmap(Mat & det, const float thresh, const int border_threshold)
{
  gsl_Expects(det.dims == 2);
  gsl_Expects(det.data != nullptr);
  Mat mask = det < thresh;
  det.setTo(0, mask);
  
  // Mat pooled = nms(det);
  // same with python version 
  // const int nms_size = 10;
  const int nms_size = 20;
  Mat pooled = nms(det, nms_size);

  vector<Point2i> pts;
  findNonZero(pooled > thresh, pts);

  vector<Point2i> non_border_pts;
  std::copy_if(pts.begin(), pts.end(),
               std::back_inserter(non_border_pts),
               [&det, &border_threshold](Point2i const & pt) {
                 return (pt.x >= border_threshold) &&
                   (pt.y >= border_threshold) &&
                   (pt.y < det.rows - border_threshold) &&
                   (pt.x < det.cols - border_threshold);
               });
  return non_border_pts;
}

// Convert a char/float mat to torch Tensor
at::Tensor matToTensor(const Mat &image)
{
  bool isChar = (image.type() & 0xF) < 2;
  vector<int64_t> dims = {image.rows, image.cols, image.channels()};
  return torch::from_blob(image.data, dims,
                          isChar ? torch::kChar : torch::kFloat).to(torch::kFloat);
}

Mat tensorToMat(const at::Tensor &tensor)
{
  gsl_Expects(tensor.ndimension() == 3 || tensor.ndimension() == 2);
  gsl_Expects(tensor.dtype() == torch::kFloat32);
  auto tensor_c = tensor.contiguous();
  auto sizes = tensor.sizes();
  if (tensor.ndimension() == 3) {
    return Mat(sizes[0], sizes[1], CV_32FC(sizes[2]), tensor_c.data_ptr());
  } else if (tensor.ndimension() == 2) {
    return Mat(sizes[0], sizes[1], CV_32F, tensor_c.data_ptr());
  } else {
      throw std::runtime_error("Cannot handle tensor dimensions other than 2 or 3");
  }
}

tuple<Mat, Mat, Mat>
  model_forward(torch::jit::script::Module model,
                const Mat& imgfloat)
{
  gsl_Expects(imgfloat.type() == CV_32FC3);
  auto input = matToTensor(imgfloat);
  // Make channel the first dimension CWH from WHC
  input = input.permute({2, 0, 1}); // WHC -> CWH
  input.unsqueeze_(0); // Make it NCWH
  torch::Device device = (*model.parameters().begin()).device();
  auto input_device = input.to(device);
  vector<torch::jit::IValue> inputs;
  inputs.push_back(input_device);
  torch::jit::IValue out = model.forward(inputs);
  auto outele = out.toTuple()->elements();
  auto heatmap_device = outele[0].toTensor();
  torch::Device cpu = torch::Device(torch::DeviceType::CPU, 0);
  auto heatmap = heatmap_device.to(cpu);
  Mat cvout = tensorToMat(heatmap[0][0]);
  auto heatmap1to3 = at::slice(heatmap[0], /*dim=*/0, /*start=*/1, /*end=*/4);
  heatmap1to3 = heatmap1to3.permute({ 1, 2, 0}); // CWH -> WHC
  Mat xyz = tensorToMat(heatmap1to3);
  Mat depth = tensorToMat(heatmap[0][4]);
  gsl_Ensures(cvout.type() == CV_32FC1);
  gsl_Ensures(xyz.type() == CV_32FC3);
  gsl_Ensures(depth.type() == CV_32FC1);
  return make_tuple(cvout.clone(), xyz.clone(), depth.clone());
}

vector<SemanticKeypoint>
mean_grouped_by_label(vector<SemanticKeypoint> const& semkp_list)
{
  unordered_map<string, vector<SemanticKeypoint>> label2idx;
  for (size_t idx = 0; idx < semkp_list.size(); idx ++) {
    SemanticKeypoint const& semkp = semkp_list[idx];
    string const& label = semkp.label;
    if (label2idx.count(label)) {
      label2idx.at(label).push_back(semkp);
    } else {
      vector<SemanticKeypoint> values({semkp});
      label2idx[label] = values;
    }
  }

  vector<SemanticKeypoint> value_uniq;
  for (auto const& keyval: label2idx) {
    vector<SemanticKeypoint> values = keyval.second;
    SemanticKeypoint value_mean = std::accumulate(values.begin(), values.end(),
                                                  SemanticKeypoint::Zero());
    float ksize = values.size();
    if (ksize) {
      value_uniq.emplace_back(value_mean / ksize);
    }
  }
  return value_uniq;
}


cv::Mat extract_patch(const cv::Mat& hm, const cv::Point2i& pt,
                      const int rx = 1,
                      const int ry = 1)
{
  gsl_Expects(pt.x - rx >= 0);
  gsl_Expects(pt.y - ry >= 0);
  gsl_Expects(pt.y + ry + 1 < hm.rows);
  gsl_Expects(pt.x + rx + 1 < hm.cols);
  gsl_Expects(hm.type() == CV_32FC1);
  cv::Rect patchrect(pt.x - rx,
                     pt.y - ry,
                     2 * ry + 1,
                     2 * rx + 1);
  cv::Mat patch = hm(patchrect);
  return patch;
}


cv::Matx22f hessian(const cv::Mat& patch)
{
  assert(patch.cols == 3);
  assert(patch.rows == 3);
  int rx = patch.cols / 2;
  int ry = patch.rows / 2;
  // Start, middle and end of patch in x and y directions
  int sx = 0, mx = rx, ex = patch.cols - 1;
  int sy = 0, my = ry, ey = patch.rows - 1;
  float h11 = patch.at<float>(my, ex) - 2 * patch.at<float>(my,mx)
    + patch.at<float>(my, sx);
  float h12 = 0.5f * (patch.at<float>(ey, ex) - patch.at<float>(ey,sx)
                      - (patch.at<float>(sy, ex) - patch.at<float>(sy, sx)));
  float h21 = 0.5f * (patch.at<float>(ey, ex) - patch.at<float>(sy, ex)
                      - (patch.at<float>(ey, sx) - patch.at<float>(sy, sx)));
  float h22 = patch.at<float>(ey, mx) - 2 * patch.at<float>(my,mx)
    + patch.at<float>(sy, mx);
  cv::Matx22f H;
  H << h11, h12, h21, h22;
  return H;
}

bool is_positive_definite(const cv::Matx22f& M)
{
  cv::Matx22f M_T = M.t();
  cv::Matx22f diff;
  cv::absdiff(M_T, M, diff);
  double diffval = cv::sum(diff)[0] / 4.0;
  assert(diffval < 1e-4); // M must be symmetric;
  // Its leading principal minors are all positive
  // https://en.wikipedia.org/wiki/Sylvester%27s_criterion
  return (M(0,0) > 0) && (cv::determinant(M) > 0);
}


cv::Matx22f SemanticKeypoint::cov_from_heatmap(cv::Mat const & hm_patch)
{
  // Assuming heatmap to the log likelihood, we take the
  // the inverse of fischer information matrix to be the covariance matrix
  cv::Matx22f cov = cv::Matx22f::eye();
  cv::Matx22f negH = -1 * hessian(hm_patch);
  if (is_positive_definite(negH)) {
    cov = negH.inv();
  }
  return cov;
}


//tuple<Points, vector<string>, vector<float>, vector<float>>
vector<SemanticKeypoint>
   find_semantic_keypoints_prob_depth(torch::jit::script::Module model,
                                      const std::string object_class,
                                      const Mat& img,
                                      const int input_res,
                                      const bool visualize,
                                      const bool unique_labels)
{
  const int ADDNL_SCALE_FACTOR = 4;
  // img2 = Crop(img, center, scale, input_res) / 256.;
  gsl_Expects(img.type() == CV_32FC3);
  Mat img_cropped = crop(img, input_res);

  Mat hm00, xyz, depth;
  tie(hm00, xyz, depth)  = model_forward(model, img_cropped);

  // same as python version 
  // const int heat_thresh = 0.15;
  const int heat_thresh = 0.3;
  auto pts = parse_keypoints_from_heatmap(hm00, heat_thresh);

  if (visualize) {
    Mat star;
    resize(hm00 * 255, star, {img_cropped.size[0], img_cropped.size[1]});
    Mat starvis;
    cvtColor(star, starvis, COLOR_GRAY2BGR);
    starvis = starvis * 0.5 + img_cropped * 255 * 0.5;
    Mat starvisimg;
    starvis.convertTo(starvisimg, CV_8UC1);
    imshow("starvis", starvisimg);
    imwrite("/tmp/starvis.png", starvisimg);
    waitKey(-1);
  }


  vector<SemanticKeypoint> semkp_list;
  for (auto const& pt: pts) {
    Point3f xyz_at = xyz.at<Point3f>(pt.y, pt.x);
    Vec3f xyz_vec{ xyz_at.x, xyz_at.y, xyz_at.z };
    semkp_list.emplace_back(pt,
                            xyz_vec,
                            depth.at<float>(pt.y, pt.x),
                            extract_patch(hm00, pt),
                            GLOBAL_OBJECT_STRUCTURE.find_semantic_part(object_class, xyz_vec));
  }
  auto pts_old_kp = convert_to_precrop(pts, {img.size[1], img.size[0]}, input_res,
                                       /*addnl_scale_factor=*/ADDNL_SCALE_FACTOR);
  for (size_t i = 0; i < pts_old_kp.size(); i++) {
    semkp_list[i].pos2d = pts_old_kp[i];
  }

  if (unique_labels) {
    semkp_list = mean_grouped_by_label(semkp_list);
  }

  return semkp_list;
}


torch::jit::script::Module
  model_load(const std::string& starmap_filepath,
             const int gpu_id)
{
  // model = torch.load(opt.loadModel)
  int device_id = gpu_id >= 0 ? gpu_id : 0;
  torch::DeviceType device_type = gpu_id >= 0 ? torch::DeviceType::CUDA : torch::DeviceType::CPU;
  torch::Device device = torch::Device(device_type, gpu_id);
  auto model = torch::jit::load(starmap_filepath,
                                /*map_location=*/device);
  return model;
}


vector<Point2i> run_starmap_on_img(const string& starmap_filepath,
                                   const string& img_filepath,
                                   const int input_res,
                                   const int gpu_id,
                                   const bool visualize)
{
    gsl_Expects(input_res > 0);

    // img = cv2.imread(opt.demo)
    const auto img = imread(img_filepath, IMREAD_COLOR);
    assert(img.type() == CV_8UC3);
    Mat imgfloat;
    img.convertTo(imgfloat, CV_32FC3, 1/255.0);

    auto model = model_load(starmap_filepath, gpu_id);

    std::string object_class = "car";
    vector<SemanticKeypoint> semkp_list =
      find_semantic_keypoints_prob_depth(model, object_class, imgfloat, input_res, visualize,
                                         /*unique_labels=*/true);

    if (visualize) {
      auto vis = img;
      visualize_keypoints(object_class, vis, semkp_list, /*draw_labels=*/true);
      imshow("vis", vis);
      imwrite("/tmp/vis.png", vis);
      waitKey(-1);
    }

    vector<Point2i> pts;
    std::transform(semkp_list.begin(), semkp_list.end(),
                   std::back_inserter(pts),
                   [](SemanticKeypoint const & semkp) -> Point2i {
                     return semkp.pos2d;
                   });

    return pts;
}

void visualize_keypoints(const std::string& object_class, Mat& vis, const vector<SemanticKeypoint>& semkp_list,
                         bool draw_labels) {

  for (auto const& semkp : semkp_list) {
    auto& pt4 = semkp.pos2d;
    auto col = GLOBAL_OBJECT_STRUCTURE.get_label_color(object_class, semkp.label);
    int radius = std::max(2, 2 * vis.rows / 40);
    circle(vis, pt4, radius + 1, Scalar(255, 255, 255), -1);
    circle(vis, pt4, radius, col, -1);
    if (draw_labels) {
        putText(vis, semkp.label, pt4,
                cv::FONT_HERSHEY_SIMPLEX,
                /*fontSize=*/std::max(0.3, 0.3 * vis.rows / 480),
                /*color=*/Scalar(255, 255, 255), /*lineThickness=*/1);
    }
  }

}

std::ostream& operator<< (std::ostream& o, const SemanticKeypoint& semkp) {
  o << "SemanticKeypoint(" << "pos2d=" << semkp.pos2d << ", "
    << "xyz=" << semkp.xyz << ","
    << "depth=" << semkp.depth << ", "
    << "hm_patch=" << semkp.hm_patch << ", "
    << "label=" << semkp.label
    << ")";
  return o;
}

}
