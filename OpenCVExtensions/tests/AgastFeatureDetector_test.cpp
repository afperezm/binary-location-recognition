/*
 * AgastFeatureDetector_test.cpp
 *
 *  Created on: Aug 20, 2013
 *      Author: andresf
 */

#include <AgastFeatureDetector.h>

#include <gtest/gtest.h>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

TEST(AgastFeatureDetector, Init) {

	cv::initModule_features2d_extensions();

	std::vector<std::string> algorithms;
	cv::Algorithm::getList(algorithms);

	std::string candidateAlgorithm = "Feature2D.AGAST";

	bool isValid = false;

	for (std::string& algorithm : algorithms) {
		isValid |= algorithm.compare(candidateAlgorithm) == 0;
	}

	ASSERT_TRUE(isValid);

}

TEST(AgastFeatureDetector, Detect) {

	cv::Mat img = cv::imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
			"AGAST");

	ASSERT_TRUE(detector->name().compare("Feature2D.AGAST") == 0);

	std::vector<cv::KeyPoint> keypoints;

	ASSERT_TRUE(keypoints.size() == 0);

	detector->detect(img, keypoints);

	ASSERT_TRUE(keypoints.size() != 0);

}
