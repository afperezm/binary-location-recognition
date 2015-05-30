/*
 * DynamicMat_test.cpp
 *
 *  Created on: Nov 3, 2013
 *      Author: andresf
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/logger.h>

#include <DynamicMat.hpp>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

TEST(DynamicMat, EmptyInstantiation) {

	vlr::Mat data;

	EXPECT_TRUE(data.empty());

}

TEST(DynamicMat, Instantiation) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	FileUtils::loadDescriptors("sift_0.bin", imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.bin");
	keysFilenames.push_back("sift_0.bin");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == imgDescriptors.rows * 2);
	EXPECT_TRUE(data.cols == imgDescriptors.cols);
	EXPECT_TRUE(data.type() == imgDescriptors.type());

}

TEST(DynamicMat, InitByCopy) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.bin");
	keysFilenames.push_back("sift_0.bin");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	vlr::Mat dataCopy(data);

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());

}

TEST(DynamicMat, InitByAssignment) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.bin");
	keysFilenames.push_back("sift_0.bin");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	vlr::Mat dataCopy = data;

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());

}

TEST(DynamicMat, RowExtraction) {

	cvflann::Logger::setDestination("row_access.log");

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	FileUtils::loadDescriptors("sift_0.bin", imgDescriptors);

	std::vector<std::string> keysFilenames(1, "sift_0.bin");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Mat extractedRow, originalRow;

	for (int i = 0; i < data.rows; i++) {

		extractedRow = cv::Mat();
		originalRow = cv::Mat();

		double mytime = cv::getTickCount();

		extractedRow = data.row(i);

		// Check that rowA is continuous though it was extracted using Mat::row
		EXPECT_TRUE(extractedRow.isContinuous());

		mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency()
				* 1000.0;

		cvflann::Logger::log(0, "%lf\n", mytime);

		originalRow = imgDescriptors.row(i % imgDescriptors.rows);

		// Check number of rows, columns and type are equal
		EXPECT_TRUE(extractedRow.rows == 1);
		EXPECT_TRUE(extractedRow.rows == originalRow.rows);
		EXPECT_TRUE(extractedRow.cols == originalRow.cols);
		EXPECT_TRUE(extractedRow.type() == originalRow.type());

		// Check row elements are equal
		for (int j = 0; j < extractedRow.cols; j++) {
			float a = originalRow.at<float>(0, j);
			float b = extractedRow.at<float>(0, j);
			EXPECT_TRUE(a == b);
		}

	}

}

//TEST(DynamicMat, WorkingPrinciple) {
//	cv::RNG rng(0xFFFFFFFF);
//	cv::Mat mat = cv::Mat::zeros(1, 5, CV_32F);
//	for (int i = 0; i < mat.rows; ++i) {
//		for (int j = 0; j < mat.cols; ++j) {
//			mat.at<float>(i, j) = (float) rng.uniform(0, 256);
//		}
//	}
//	std::cout << mat.at<float>(0, 0) << std::endl;
//	std::cout << mat.at<float>(0, 1) << std::endl;
//	uchar* p_mat = mat.data;
//	std::vector<char> value(p_mat, p_mat + mat.step * mat.rows);
//	float* p_vector = reinterpret_cast<float*>(value.data());
//	std::cout << *p_vector << std::endl;
//	std::cout << *(p_vector + 1) << std::endl;
//}
