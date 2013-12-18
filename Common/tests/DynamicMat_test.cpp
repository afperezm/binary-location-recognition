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

	DynamicMat data;

	EXPECT_TRUE(data.empty());
	EXPECT_TRUE(data.getDescriptorsIndex().size() == 0);
	EXPECT_TRUE(data.getDescriptorsFilenames().size() == 0);

}

TEST(DynamicMat, Instantiation) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	FileUtils::loadDescriptors("sift_0.yaml.gz", imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_0.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == imgDescriptors.rows * 2);
	EXPECT_TRUE(data.cols == imgDescriptors.cols);
	EXPECT_TRUE(data.type() == imgDescriptors.type());
	EXPECT_TRUE(data.getDescriptorsFilenames().size() == keysFilenames.size());

}

TEST(DynamicMat, InitByCopy) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_0.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	DynamicMat dataCopy(data);

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());
	EXPECT_TRUE(
			data.getDescriptorsIndex().size()
					== dataCopy.getDescriptorsIndex().size());
	EXPECT_TRUE(
			data.getDescriptorsFilenames().size()
					== dataCopy.getDescriptorsFilenames().size());

}

TEST(DynamicMat, InitByAssignment) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_0.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	DynamicMat dataCopy = data;

	EXPECT_FALSE(data.empty());
	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());
	EXPECT_TRUE(
			data.getDescriptorsIndex().size()
					== dataCopy.getDescriptorsIndex().size());
	EXPECT_TRUE(
			data.getDescriptorsFilenames().size()
					== dataCopy.getDescriptorsFilenames().size());

}

TEST(DynamicMat, RowExtraction) {

	cvflann::Logger::setDestination("row_access.log");

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	FileUtils::loadDescriptors("sift_0.yaml.gz", imgDescriptors);

	std::vector<std::string> keysFilenames(10, "sift_0.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Mat extractedRow, originalRow;

	for (int i = 0; i < data.rows; i++) {

		extractedRow = cv::Mat();
		originalRow = cv::Mat();

		double mytime = cv::getTickCount();

		extractedRow = data.row(i);

		// Check that rowA is continuous thought it was extracted using Mat::row
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
			EXPECT_TRUE(extractedRow.at<float>(0, j) == originalRow.at<float>(0, j));
		}
	}

}

TEST(DynamicMat, Stress) {
	std::string root = "/home/andresf/oxford_buildings_dataset/";
	std::vector<std::string> descriptorsList;
	FileUtils::loadList("list_noqueries_distract.txt", descriptorsList);
	DynamicMat data(descriptorsList);
	for (int i = 0; i < data.rows; ++i) {
		ASSERT_TRUE(data.getMemoryCount() <= data.MAX_MEM);
	}
}
