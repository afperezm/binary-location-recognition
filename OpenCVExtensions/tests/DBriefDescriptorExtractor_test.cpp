/*
 * DBriefDescriptorExtractor_test.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: andresf
 */

#include <DBriefDescriptorExtractor.h>

#include <gtest/gtest.h>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

TEST(DBriefFeatureExtractor,Init) {

	cv::initModule_features2d_extensions();

	std::vector<std::string> algorithms;
	cv::Algorithm::getList(algorithms);

	std::string candidateAlgorithm = "Feature2D.DBRIEF";

	bool isValid = false;

	for (std::string& algorithm : algorithms) {
		isValid |= algorithm.compare(candidateAlgorithm) == 0;
	}

	ASSERT_TRUE(isValid);

}

TEST(DBriefFeatureExtractor,Extract) {

	cv::Mat img = cv::imread("test.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create("FAST");

	std::vector<cv::KeyPoint> keypoints;
	detector->detect(img, keypoints);

	cv::Ptr<cv::DescriptorExtractor> extractor =
			cv::DescriptorExtractor::create("DBRIEF");

	cv::Mat descriptors;

	ASSERT_TRUE(descriptors.empty());

	extractor->compute(img, keypoints, descriptors);

	ASSERT_FALSE(descriptors.empty());

}
