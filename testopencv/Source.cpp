#include <iostream>
#include <utility>
#include <string>
#include "segm/msImageProcessor.h"
#include "edge/BgEdgeDetect.h"
#include "edge/BgImage.h"
#include <opencv2/opencv.hpp>

using namespace std;
int main() {
	int height, width;

	unsigned char* inputHsv_;
	unsigned char* inputHue_;
	float* weightMap_;
	unsigned char* filtImage_;
	unsigned char* segmImage_;
	//store the output edges and boundaries
	int* edges_, numEdges_;
	int* boundaries_, numBoundaries_;
	//parameters for mean shift
	int sigmaS;     //spatial bandwidth
	float sigmaR;       //range bandwidth
	int minRegion;  //area of the smallest objects to consider
	//parameters for synergistic segmentation
	int gradWindRad; //Gradient Window Radius
	float threshold; //Edge Strength Threshold [0,1]
	float mixture;   //Mixture Parameter [0,1]

	sigmaS = 7;
	sigmaR = 6.5;
	minRegion = 20;
	gradWindRad = 2;
	threshold = 0.3F;
	mixture = 0.2F;


	// Load the image
	cv::Mat src_bgr = cv::imread("rose.png"); // Reads as BGR
	cv::Mat src;
	cv::cvtColor(src_bgr, src, cv::COLOR_BGR2RGB);
	if (!src.data) {
		return -1;
	}

	// Convert to char buffer
	height = src.rows;
	width = src.cols;
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
	//compute the weight map
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

	//uchar* defined_image = new uchar[height * width * 3];
	//iProc.GetResults(defined_image);
	//cv::Mat defined(height, width, CV_8UC4, defined_image);

	// Show source image
	cv::imshow("Source Image", src);
	cv::waitKey(0);
	return 0;
}