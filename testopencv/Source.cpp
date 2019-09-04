#include <ctime>
#include <iostream>
#include <utility>
#include <string>

#include <opencv2/opencv.hpp>

#include "segm/msImageProcessor.h"
#include "edge/BgEdgeDetect.h"
#include "edge/BgImage.h"

using namespace std;

int main() {
	std::clock_t start;
	int duration;

	//parameters for mean shift
	const int sigmaS = 7;           // spatial bandwidth
	const float sigmaR = 6.5;       // range bandwidth
	const int minRegion = 20;       // area of the smallest objects to consider
	//parameters for synergistic segmentation
	const int gradWindRad = 2;      // gradient window radius
	const float threshold = 0.3F;   // Edge strength threshold [0,1]
	const float mixture = 0.2F;     // mixture parameter [0,1]
	const SpeedUpLevel speedup = SpeedUpLevel::MED_SPEEDUP;

	// Load the image as BGR (src_bgr) then convert to RGB (src)
	const cv::Mat src_bgr = cv::imread("rose.png");
	cv::Mat src;
	cv::cvtColor(src_bgr, src, cv::COLOR_BGR2RGB);

	// Compute edges
	const int height = src.rows;
	const int width = src.cols;
	vector<float> gradMap(width * height);
	vector<float> confMap(width * height);
	vector<float> weightMap(width * height);
	BgEdgeDetect edgeDetector(gradWindRad);
	BgImage bgImage;
	bgImage.SetImage(src.data, width, height, true);
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

	// Filter and fuse regions
	msImageProcessor iProc;
	//printf("Filtering and fusing: ");
	//start = clock();
	iProc.DefineImage(src.data, COLOR, height, width);
	assert(!iProc.ErrorStatus);
	iProc.SetWeightMap(weightMap.data(), threshold);
	assert(!iProc.ErrorStatus);
	iProc.Filter(sigmaS, sigmaR, speedup);
	assert(!iProc.ErrorStatus);
    iProc.FuseRegions(sigmaR, minRegion);
	assert(!iProc.ErrorStatus);
	//duration = (clock() - start) * 1000 / (double)CLOCKS_PER_SEC;
	//printf("%d ms\n", duration);

	// The memory for the following arrays is managed by msImageProcessor.
	// width * height sized array mapping each pixel to a label identifying a
	// region (i.e. a segment). Stored in row major order, as usual.
	int* labels;
	// regionCount * 3 sized array. Stores the color value for each region in
	// LUV space sequentially (for each region, stores 3 floats for L, U, and V).
	// e.g. the eighth region is at indices 3*8 + 0, 3*8 + 1, and 3*8 + 2.
	float* modes;
	// regionCount sized array mapping region label to the number of points in it.
	int* modePointCounts;
	const int regionCount = iProc.GetRegions(&labels, &modes, &modePointCounts);

	// Can be uncommented if you want to visualize the final segmented image.
	//vector<float> finalImageLuvVec(height * width * 3);
	//for (int i = 0; i < height * width; ++i) {
	//	const int label = labels[i];
	//	for (int dim = 0; dim < 3; ++dim) {
	//		finalImageLuvVec[i * 3 + dim] = modes[labels[i] * 3 + dim];
	//	}
	//}
	//cv::Mat finalImageLuv(height, width, CV_32FC3, finalImageLuvVec.data());
	//cv::Mat finalImageBgr;
	//cv::cvtColor(finalImageLuv, finalImageBgr, cv::COLOR_Luv2BGR);
	//cv::imshow("Image", finalImageBgr);
	//cv::waitKey(0);

	return 0;
}