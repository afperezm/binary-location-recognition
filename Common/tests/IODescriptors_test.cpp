/*
 * IODescriptors_test.cpp
 *
 *  Created on: Nov 21, 2013
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>
#include <opencv2/flann/logger.h>

#include <FileUtils.hpp>

TEST(IODescriptors, LoadSave) {

	cv::Mat original;
	FileUtils::loadDescriptors("sift_0.yaml.gz", original);
	FileUtils::saveDescriptors("sift_desc_tmp.yaml.gz", original);

	cv::Mat loaded;
	FileUtils::loadDescriptors("sift_desc_tmp.yaml.gz", loaded);

	// Check number of rows, columns and type are equal
	EXPECT_TRUE(original.rows == loaded.rows);
	EXPECT_TRUE(original.cols == loaded.cols);
	EXPECT_TRUE(original.type() == loaded.type());

	// Check elements are equal
	for (int i = 0; i < original.rows; i++) {
		for (int j = 0; j < original.cols; j++) {
			EXPECT_TRUE(original.at<float>(i, j) == loaded.at<float>(i, j));
		}
	}

}

TEST(IODescriptors, LoadStress) {

	cvflann::Logger::setDestination("load_times.log");

	cv::Mat descriptors;

	for (size_t i = 0; i < 50; i++) {
		descriptors = cv::Mat();

		double mytime = cv::getTickCount();
		FileUtils::loadDescriptors("sift_0.yaml.gz", descriptors);
		mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency()
				* 1000.0;

		cvflann::Logger::log(0, "%lf\n", mytime);
	}
}
