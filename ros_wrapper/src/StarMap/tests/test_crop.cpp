#include <sstream>
#include <stdexcept> // std::runtime_error
#include "gtest/gtest.h" // TEST()
#include "starmap/starmap.h" // starmap::

#include "opencv2/opencv.hpp" // cv::*
#include "opencv2/imgcodecs.hpp" // cv::imread, cv::imwrite

class FileNotFoundError : std::runtime_error {
public:
  FileNotFoundError(const std::string& what_arg ) : std::runtime_error( what_arg ) { }
  FileNotFoundError(const char* what_arg ) : std::runtime_error( what_arg ) { }
};

cv::Mat imread_img(const cv::Size img_shape,
                   const std::string fname = "tests/data/lena512.pgm") {
  cv::Mat outimg;
  cv::Mat inimg = cv::imread(fname, cv::IMREAD_UNCHANGED);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(fname);
  cv::resize(inimg, outimg, img_shape);
  return outimg;
}


/**
 * @param img_shape: (height, width)
 */
bool _test_crop_lenna(const int rows,
                      const int cols,
                      const int desired_side = 256,
                      const int rot = 0) {
  cv::Size imgshape(rows, cols);
  auto img = imread_img(imgshape);
  auto cropped = starmap::crop(img, desired_side);
  std::ostringstream filepath;
  filepath << "tests/data/test-crop-lenna-" << rows << "-" << cols << ".pgm";
  // cv::imwrite(filepath.str(), cropped);
  cv::Mat expected_img = cv::imread(filepath.str(), cv::IMREAD_UNCHANGED);
  cv::Mat diff = cropped != expected_img;
  return cv::countNonZero(diff) == 0;
}


TEST(CropTest, HandlesFatBigBig) {
  ASSERT_TRUE(_test_crop_lenna(301, 401)) << "Images do not match";
}


TEST(CropTest, HandlesFatSmallBig) {
  ASSERT_TRUE(_test_crop_lenna(101, 401)) << "Images do not match";
}

TEST(CropTest, HandlesFatSmallSmall) {
  ASSERT_TRUE(_test_crop_lenna(101, 201)) << "Images do not match";
}

TEST(CropTest, HandlesTallSmallSmall) {
  ASSERT_TRUE(_test_crop_lenna(201, 101)) << "Images do not match";
}

TEST(CropTest, HandlesTallBigSmall) {
  ASSERT_TRUE(_test_crop_lenna(401, 101)) << "Images do not match";
}

TEST(CropTest, HandlesTallBigBig) {
  ASSERT_TRUE(_test_crop_lenna(401, 301)) << "Images do not match";
}

TEST(CropTest, CarCropTest) {
  constexpr bool DEBUG = false;
  const std::string infname("tests/data/car.jpg");
  const std::string expectedfname("tests/data/car-cropped.jpg");
  const int desired_side = 256;
  cv::Mat inimg = cv::imread(infname, cv::IMREAD_COLOR);
  if (inimg.data == nullptr)
    throw new FileNotFoundError(infname);
  auto cropped = starmap::crop(inimg, desired_side);
  cv::Mat expected_img = cv::imread(expectedfname, cv::IMREAD_COLOR);
  if (DEBUG) {
    cv::imshow("cropped", cropped);
    cv::imshow("expected_img", expected_img);
  }
  cv::Mat cropped_gray, expected_img_gray;
  cv::cvtColor(cropped,  cropped_gray, cv::COLOR_BGR2GRAY);
  cv::cvtColor(expected_img, expected_img_gray, cv::COLOR_BGR2GRAY);
  cv::Mat diff;
  cv::absdiff(cropped_gray, expected_img_gray, diff);

  if (DEBUG) {
    cv::imshow("diff", diff);
    cv::waitKey(-1);
  }
  double diffval = cv::sum(diff)[0];
  ASSERT_TRUE( diffval < 0.02 * 255 * desired_side * desired_side) <<
    "diffval: " << diffval << " < exp: " << 0.02 * 255 * desired_side * desired_side;
}
