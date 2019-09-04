#include <iostream>
#include <utility>
#include <string>
#include "segm/msImageProcessor.h"
#include "edge/BgEdgeDetect.h"
#include "edge/BgImage.h"
#include <opencv2/opencv.hpp>

using namespace std;
int main() {
	//parameters for mean shift
	const int sigmaS = 7;           //spatial bandwidth
	const float sigmaR = 6.5;       //range bandwidth
	const int minRegion = 20;       //area of the smallest objects to consider
	//parameters for synergistic segmentation
	const int gradWindRad = 2;      //Gradient Window Radius
	const float threshold = 0.3F;   //Edge Strength Threshold [0,1]
	const float mixture = 0.2F;     //Mixture Parameter [0,1]
	const SpeedUpLevel speedup = MED_SPEEDUP;

	// Load the image (as BGR) then convert to RGB
	const cv::Mat src_bgr = cv::imread("rose.png");
	cv::Mat src;
	cv::cvtColor(src_bgr, src, cv::COLOR_BGR2RGB);
	assert(src.data != nullptr);

	// Convert to char buffer
	const int height = src.rows;
	const int width = src.cols;
	uchar* image_buf;
	if (src.isContinuous()) {
		image_buf = src.data;
	}
	else {
		cerr << "Image not continuous. TODO" << endl;
		return 1;
	}

	vector<float> gradMap(width * height);
	vector<float> confMap(width * height);
	vector<float> weightMap(width * height);
	BgEdgeDetect edgeDetector(gradWindRad);
	BgImage bgImage;
	bgImage.SetImage(image_buf, width, height, true);
	edgeDetector.ComputeEdgeInfo(&bgImage, confMap.data(), gradMap.data());

	// Compute weight map
	for (int i = 0; i < width * height; i++) {
		if (gradMap[i] > 0.02) {
			weightMap[i] = mixture * gradMap[i] + (1 - mixture) * confMap[i];
		}
		else {
			weightMap[i] = 0;
		}
	}

	msImageProcessor iProc;
	iProc.DefineImage(image_buf, COLOR, height, width);
	assert(!iProc.ErrorStatus);
	iProc.SetWeightMap(weightMap.data(), threshold);
	assert(!iProc.ErrorStatus);
	iProc.Filter(sigmaS, sigmaR, speedup);
	assert(!iProc.ErrorStatus);
    iProc.FuseRegions(sigmaR, minRegion);
	assert(!iProc.ErrorStatus);

	int* labels;
	float* modes;
	int* modePointCounts;
	const int regionCount = iProc.GetRegions(&labels, &modes, &modePointCounts);

	// Can be uncommented if you want to visualize the final image
	//vector<float> final_image_luv_vec(height * width * 3);
	//for (int i = 0; i < height * width; ++i) {
	//	const int label = labels[i];
	//	for (int dim = 0; dim < 3; ++dim) {
	//		final_image_luv_vec[i * 3 + dim] = modes[labels[i] * 3 + dim];
	//	}
	//}
	//cv::Mat final_image_luv(height, width, CV_32FC3, final_image_luv_vec.data());
	//cv::Mat final_image_bgr;
	//cv::cvtColor(final_image_luv, final_image_bgr, cv::COLOR_Luv2BGR);
	//cv::imshow("Source Image", final_image_bgr);

	cv::waitKey(0);
	return 0;
}