#include <sstream>
#include <stdexcept> // std::runtime_error
#include "gtest/gtest.h" // TEST()
#include "starmap/starmap.h" // starmap::*
#include "torch/torch.h" // torch::*
#include "torch/script.h" // torch::*
#include "boost/filesystem.hpp" // boost::filesystem::*

#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/imgcodecs.hpp" // cv::imread, cv::imwrite

namespace tjs = torch::jit::script;
namespace bfs = boost::filesystem;

class FileNotFoundError : std::runtime_error {
public:
  FileNotFoundError(const std::string& what_arg ) : std::runtime_error( what_arg ) { }
  FileNotFoundError(const char* what_arg ) : std::runtime_error( what_arg ) { }
};

cv::Mat safe_cv2_imread(const std::string fname = "tests/data/lena512.pgm") {
  cv::Mat inimg = cv::imread(fname, cv::IMREAD_UNCHANGED);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(fname);
  return inimg;
}

TEST(HmParser, Nms) {
    cv::Mat img = safe_cv2_imread();
    cv::Mat det;
    img.convertTo(det, CV_32F);
    det = det / 255;
    cv::Mat pool =  starmap::nms(det);
    cv::Mat pooli8;
    pool.convertTo(pooli8, CV_8U, 255);
    cv::Mat expected = safe_cv2_imread("tests/data/test-lenna-nms-out.pgm");
    ASSERT_TRUE(pooli8.size == expected.size);
    cv::Mat diff = pooli8 != expected;
    ASSERT_TRUE(cv::countNonZero(diff) == 0) << "countNonZero: " << cv::countNonZero(diff);
}

TEST(HmParser, parseHeatmap) {
    cv::Mat hm;
    safe_cv2_imread().convertTo(hm, CV_32F, 1/ 255.);
    auto pts = starmap::parse_keypoints_from_heatmap(hm);

    // Serialize using opencv
    cv::FileStorage fs("tests/data/test-lenna-parseHeatmap-out.cv2.yaml",
                       cv::FileStorage::READ);
    auto expected_pts = fs.getFirstTopLevelNode().mat();
    ASSERT_EQ(expected_pts.size[1], pts.size());
    for (int i = 0; i < expected_pts.size[1]; ++i) {
        ASSERT_EQ(expected_pts.at<int>(0, i), pts[i].y) << "Fail for i = " << i;
        ASSERT_EQ(expected_pts.at<int>(1, i), pts[i].x) << "Fail for i = " << i;
    }
}

TEST(HmParser, model_forward_car) {
  const std::string starmap_filepath("models/model_cpu-jit.pth");
  if (! bfs::exists(starmap_filepath))
    return;

  const std::string croppedimg("tests/data/car-cropped.jpg");
  const cv::FileStorage fs("tests/data/car-hm00.cv2.yaml",
                           cv::FileStorage::READ);
  cv::Mat inimg = cv::imread(croppedimg, cv::IMREAD_COLOR);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(croppedimg);
  cv::Mat imgfloat;
  inimg.convertTo(imgfloat, CV_32FC3, 1/255.0);
  auto model = torch::jit::load(starmap_filepath);
  cv::Mat hm00, xyz, depth;
  std::tie(hm00, xyz, depth) = starmap::model_forward(model, imgfloat);
  auto expected_hm00 = fs["hm00"].mat();
  ASSERT_EQ(hm00.size, expected_hm00.size);
  cv::Mat diff;
  cv::absdiff(hm00, expected_hm00, diff);
  double diffval = cv::sum(diff)[0];
  double maxdiff = expected_hm00.size[0] * expected_hm00.size[1];
  ASSERT_TRUE( diffval < 0.02 * maxdiff) <<
    "diffval: " << diffval << " < exp: " << 0.02 * maxdiff;
}

TEST(HmParser, parseHeatmap_car) {
  cv::FileStorage fs("tests/data/car-hm00.cv2.yaml",
                     cv::FileStorage::READ);
  cv::FileStorage fs2("tests/data/car-pts.cv2.yaml",
                      cv::FileStorage::READ);
  auto hm0 = fs["hm00"].mat();
  // std::cerr << cv::format(hm0, cv::Formatter::FMT_PYTHON) << "\n";
  auto pts = starmap::parse_keypoints_from_heatmap(hm0);

  // Serialize using opencv
  auto expected_pts = fs2["pts"].mat();
  ASSERT_EQ(expected_pts.size[1], pts.size());
  for (int i = 0; i < expected_pts.size[1]; ++i) {
    ASSERT_EQ(expected_pts.at<int>(0, i), pts[i].y) << "Fail y for i = " << i;
    ASSERT_EQ(expected_pts.at<int>(1, i), pts[i].x) << "Fail x for i = " << i;
  }
}

TEST(CarStructure, exact_match) {
  starmap::CarStructure car;
  auto partname = car.find_semantic_part({0.16347528,  0.2507412 , -0.07981754});
  ASSERT_EQ(partname, "right_back_wheel");

  partname = car.find_semantic_part({-0.09472257, -0.07266671,  0.10419698});
  ASSERT_EQ(partname, "upper_left_windshield");
}

TEST(CarStructure, approx_match) {
  cv::Matx<float, 3, 1> pt {-0.09472257, -0.07266671,  0.10419698};
  auto ptrn = pt + cv::Matx<float, 3, 1>::randn(0, 0.0001);
  auto partname = starmap::GLOBAL_CAR_STRUCTURE.find_semantic_part(pt.col(0));
  ASSERT_EQ(partname, "upper_left_windshield");
}
