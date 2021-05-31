/*
 * @Descripttion: Main function to boost orcvio.
 * @Author: Xiaochen Qiu
 */


#include <iostream>

#include "utils/DataReader.hpp"

#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"

#include "sensors/ImageData.hpp"

#include "orcvio/image_processor.h"
#include "orcvio/orcvio.h"

#include "Eigen/Dense"

#include "visualization/visualize.hpp"

using namespace std;


int main(int argc, char **argv) {

#ifdef VS_DEBUG_MODE
	// for VS debug 
	char imu_data_dir[] = "/media/erl/disk2/euroc/MH_01_easy/mav0/imu0/data.csv";
	char img_info_dir[] = "/media/erl/disk2/euroc/MH_01_easy/mav0/cam0/data.csv";
	char img_data_dir[] = "/media/erl/disk2/euroc/MH_01_easy/mav0/cam0/data";
	char config_dir[] = "/home/erl/Workspace/orcvio_cpp/config/euroc.yaml";
#else
    if(argc != 5) {
		cerr << endl << "Usage: ./orcvio path_to_imu/data.csv path_to_cam0/data.csv path_to_cam0/data config_file_path" << endl;
        return 1;
    }
#endif

	// Read sensors
    vector<orcvio::ImuData> allImuData;
    vector<orcvio::ImgInfo> allImgInfo;

#ifdef VS_DEBUG_MODE
    orcvio::loadImuFile(imu_data_dir, allImuData);
    orcvio::loadImageList(img_info_dir, allImgInfo);
#else
    orcvio::loadImuFile(argv[1], allImuData);
    orcvio::loadImageList(argv[2], allImgInfo);
#endif

    // Path of config file
#ifdef VS_DEBUG_MODE
    std::string config_file(config_dir);
#else 
    std::string config_file(argv[4]);
#endif

	// Initialize image processer.
	orcvio::ImageProcessorPtr ImgProcesser;
	ImgProcesser.reset(new orcvio::ImageProcessor(config_file));
	if (!ImgProcesser->initialize()) {
		cerr << "Image Processer initialization failed!" << endl;
		return 1;
    }

	// Initialize estimator.
	orcvio::OrcVIOPtr Estimator;
	Estimator.reset(new orcvio::OrcVIO(config_file));
	if (!Estimator->initialize()) {
		cerr << "Estimator initialization failed!" << endl;
		return 1;
	}

	// Initialize visualizer
	pangolin::CreateWindowAndBind("OrcVIO Viewer",1024,768);
    glEnable(GL_DEPTH_TEST);	// 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_BLEND);			// Issue specific OpenGl we might need
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowActivePoints("menu.Show Active Points",true,true);
    pangolin::Var<bool> menuShowStablePoints("menu.Show Stable Points",true,true);
    pangolin::Var<bool> menuShowSlideWindow("menu.Show Slide Window",true,true);
    pangolin::OpenGlRenderState s_cam(						// Define Camera Render Object (for view / scene browsing)
                pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0,3.5,9, 0,0,0, 0,-1,0)
                );
    pangolin::View& d_cam = pangolin::CreateDisplay()		// Add named OpenGL viewport to window and provide 3D Handler
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
	pangolin::View& d_image = pangolin::CreateDisplay()		// Add named OpenGL viewport to image and provide 3D Handler
            .SetBounds(pangolin::Attach::Pix(0),1.0f,pangolin::Attach::Pix(175),1/3.5f,752.0/480)
            .SetLock(pangolin::LockLeft, pangolin::LockBottom);
	bool bFollow = true;
	pangolin::OpenGlMatrix Tbw_pgl;
	std::vector<pangolin::OpenGlMatrix> vPoses;
	std::vector<pangolin::OpenGlMatrix> vPoses_SW;
	std::map<orcvio::FeatureIDType,Eigen::Vector3d> mActiveMapPoints;
	std::map<orcvio::FeatureIDType,Eigen::Vector3d> mStableMapPoints;

	// Main loop
	int k = 0;
	std::vector<orcvio::ImuData> imu_msg_buffer;
    for(size_t j=0; j<allImgInfo.size(); j++) {
		// get img
		string temp = allImgInfo[j].imgName.substr(0,allImgInfo[j].imgName.size()-1);
		char *fullPath = new char[100];
    	memset(fullPath,0,100);

#ifdef VS_DEBUG_MODE
		sprintf(fullPath,"%s/%s", img_data_dir, temp.c_str());
#else 
		sprintf(fullPath,"%s/%s", argv[3],temp.c_str());
#endif 

		orcvio::ImageDataPtr imgPtr(new orcvio::ImgData);
		imgPtr->timeStampToSec = allImgInfo[j].timeStampToSec;
		imgPtr->image = cv::imread(fullPath,0);

		// get imus
		while (allImuData[k].timeStampToSec-imgPtr->timeStampToSec < 0.05 && k<allImuData.size()) {
			imu_msg_buffer.push_back(
				orcvio::ImuData(allImuData[k].timeStampToSec, allImuData[k].angular_velocity, allImuData[k].linear_acceleration));
			++k;
		}

		// process
		orcvio::MonoCameraMeasurementPtr features = new orcvio::MonoCameraMeasurement;
		int64 start_time_fe = cv::getTickCount();
		bool bProcess = ImgProcesser->processImage(imgPtr, imu_msg_buffer, features);				// process image
		int64 end_time_fe = cv::getTickCount();
		double processing_time_fe = (end_time_fe - start_time_fe)/cv::getTickFrequency();
		double processing_time_be;
		bool bPubOdo = false;
		if (bProcess) {
			int64  start_time_be = cv::getTickCount();
			bPubOdo = Estimator->processFeatures(features, imu_msg_buffer);							// state estimation
			int64 end_time_be = cv::getTickCount();
			processing_time_be = (end_time_be - start_time_be)/cv::getTickFrequency();
		}
		if (bPubOdo) {	// Visualization
			// draw 3D
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			orcvio::visualize::GetCurrentOpenGLPoseMatrix(Tbw_pgl, Estimator->getTbw());
			if(menuFollowCamera && bFollow) {
				s_cam.Follow(Tbw_pgl);
			}
			else if(menuFollowCamera && !bFollow) {
				s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,3.5,9, 0,0,0, 0,-1,0));
				s_cam.Follow(Tbw_pgl);
				bFollow = true;
			}
			else if(!menuFollowCamera && bFollow) {
				bFollow = false;
			}
			d_cam.Activate(s_cam);
			glClearColor(0.0f,0.0f,0.0f,1.0f);
			orcvio::visualize::DrawCurrentPose(Tbw_pgl);
			orcvio::visualize::DrawKeyFrames(vPoses);
			if (menuShowSlideWindow) {
				vector<Eigen::Isometry3d> swPoses;
				Estimator->getSwPoses(swPoses);
				orcvio::visualize::GetSwOpenGLPoseMatrices(vPoses_SW, swPoses);
				orcvio::visualize::DrawSlideWindow(vPoses_SW);
			}
			if (menuShowActivePoints) {
				Estimator->getActiveMapPointPositions(mActiveMapPoints);
				orcvio::visualize::DrawActiveMapPoints(mActiveMapPoints);
				mActiveMapPoints.clear();
			}
			if (menuShowStablePoints) {
				Estimator->getStableMapPointPositions(mStableMapPoints);
				orcvio::visualize::DrawStableMapPoints(mStableMapPoints);
			}

			// draw tracking image
			cv::Mat img4Show;
			ImgProcesser->getVisualImg().copyTo(img4Show);
			stringstream ss;
			ss << "fps: " << 1.0/(processing_time_fe+processing_time_be) << " HZ";
			string msg = ss.str();
			int baseLine=0;
			cv::Size textSize = cv::getTextSize(msg, 0, 0.01, 10, &baseLine);
			cv::Point textOrigin(2*baseLine,textSize.height+4*baseLine);
			putText(img4Show, msg, textOrigin, 1, 2, cv::Scalar(0,165,255), 2);
			pangolin::GlTexture imageTexture(img4Show.cols,img4Show.rows);;
			imageTexture.Upload(img4Show.data,GL_BGR,GL_UNSIGNED_BYTE);
			d_image.Activate();
			glColor3f(1.0,1.0,1.0);
			imageTexture.RenderToViewportFlipY();

			pangolin::FinishFrame();
			
			vPoses.push_back(Tbw_pgl);
		}
	}

	cout << "Totally " << mStableMapPoints.size() << " SLAM points" << endl;

	// keep the final scene in pangolin
	while (!pangolin::ShouldQuit()) {
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		if(menuFollowCamera && bFollow) {
			s_cam.Follow(Tbw_pgl);
		}
		else if(menuFollowCamera && !bFollow) {
			s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,3.5,9, 0,0,0, 0,-1,0));
			s_cam.Follow(Tbw_pgl);
			bFollow = true;
		}
		else if(!menuFollowCamera && bFollow) {
			bFollow = false;
		}
		d_cam.Activate(s_cam);
		glClearColor(0.0f,0.0f,0.0f,1.0f);
		orcvio::visualize::DrawCurrentPose(Tbw_pgl);
		orcvio::visualize::DrawKeyFrames(vPoses);
		if (menuShowSlideWindow) {
			orcvio::visualize::DrawSlideWindow(vPoses_SW);
		}
		if (menuShowStablePoints) {
			orcvio::visualize::DrawStableMapPoints(mStableMapPoints);
		}
		pangolin::FinishFrame();
	}

    return 0;
}