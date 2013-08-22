//============================================================================
// Name        : MediaEval-PlacingTask.cpp
// Author      : Andrés Pérez
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <vector>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <AgastFeatureDetector.h>
#include <opencv2/core/internal.hpp>

using cv::Mat;
using std::vector;

void printParams(cv::Ptr<cv::Algorithm> algorithm);

double mytime;

namespace cv {
CV_INIT_ALGORITHM(AgastFeatureDetector, "Feature2D.AGAST",
		obj.info()->addParam(obj, "threshold", obj.threshold); obj.info()->addParam(obj, "nonmaxsuppression", obj.nonmaxsuppression); obj.info()->addParam(obj, "type", obj.type))
;
}

int main(int argc, char **argv) {

	if (argc != 2) {
		printf("\n");
		printf("Usage: %s <img1>", argv[0]);
		printf("\n\n");
		return EXIT_FAILURE;
	}

	// Initiating nonfree module, it's necessary for using SIFT and SURF
	cv::initModule_nonfree();

	printf("-- Loading image [%s]\n", argv[1]);

	mytime = (double) cv::getTickCount();
	Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	if (img_1.empty()) {
		fprintf(stderr, "-- Error reading image [%s]\n", argv[1]);
	}
	printf("-- Image loaded in [%lf] ms\n", mytime);

	// Step 1/4: detect keypoints using FAST or AGAST
	std::vector<cv::KeyPoint> keypoints_1;
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
			"AGAST");
	printParams(detector);

	mytime = cv::getTickCount();
	detector->detect(img_1, keypoints_1);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("-- Detected [%zu] keypoints in [%lf] ms\n", keypoints_1.size(),
			mytime);

//	for (cv::KeyPoint k : keypoints_1) {
//		printf("angle=[%f] octave=[%d] response=[%f] size=[%f] x=[%f] y=[%f] class_id=[%d]\n",
//				k.angle, k.octave, k.response, k.size, k.pt.x, k.pt.y, k.class_id);
//	}

// Step 2/4: extract descriptors using BRIEF or DBRIEF
	cv::Ptr<cv::DescriptorExtractor> extractor =
			cv::DescriptorExtractor::create("BRIEF");

	Mat descriptors_1;

	mytime = cv::getTickCount();
	extractor->compute(img_1, keypoints_1, descriptors_1);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	// Notice that number of keypoints might be reduced due to border effect
	printf(
			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
			descriptors_1.rows, descriptors_1.cols,
			descriptors_1.type() == CV_8U ? "binary" : "real-valued", mytime);

// Step 3/4: show keypoints

	cv::drawKeypoints(img_1, keypoints_1, img_1, cv::Scalar::all(-1));
	cv::namedWindow("Image keypoints", CV_WINDOW_NORMAL);
	cv::imshow("Image keypoints", img_1);

	cv::waitKey(0);

	// Step 4/4: save descriptors into a file for later use

	return EXIT_SUCCESS;
}

void printParams(cv::Ptr<cv::Algorithm> algorithm) {
	std::vector<std::string> parameters;
	algorithm->getParams(parameters);

	for (int i = 0; i < (int) parameters.size(); i++) {
		std::string param = parameters[i];
		int type = algorithm->paramType(param);
		std::string helpText = algorithm->paramHelp(param);
		std::string typeText;

		switch (type) {
		case cv::Param::BOOLEAN:
			typeText = "bool";
			break;
		case cv::Param::INT:
			typeText = "int";
			break;
		case cv::Param::REAL:
			typeText = "real (double)";
			break;
		case cv::Param::STRING:
			typeText = "string";
			break;
		case cv::Param::MAT:
			typeText = "Mat";
			break;
		case cv::Param::ALGORITHM:
			typeText = "Algorithm";
			break;
		case cv::Param::MAT_VECTOR:
			typeText = "Mat vector";
			break;
		}
		std::cout << "Parameter '" << param << "' type=" << typeText << " help="
				<< helpText << std::endl;
	}
}
