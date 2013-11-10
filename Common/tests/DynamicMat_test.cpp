/*
 * FunctionUtils.cpp
 *
 *  Created on: Nov 3, 2013
 *      Author: andresf
 */

#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <opencv2/features2d/features2d.hpp>

#include <DynamicMat.hpp>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>

TEST(DynamicMat, EmptyInstantiation) {

	DynamicMat data;

	EXPECT_TRUE(data.rows == 0);
	EXPECT_TRUE(data.cols == 0);
	EXPECT_TRUE(data.type() == -1);
	EXPECT_TRUE(data.getDescriptorsIndices().size() == 0);
	EXPECT_TRUE(data.getKeysFilenames().size() == 0);

}

TEST(DynamicMat, Instantiation) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;
	std::vector<cv::KeyPoint> imgKeypoints;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		imgDescriptors = cv::Mat();
		imgKeypoints.clear();
		// Load keypoints and descriptors
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);
		// Check that keypoints and descriptors have same length
		for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
			image img;
			img.imgIdx = imgIdx;
			img.startIdx = descCount;
			descriptorsIndices.push_back(img);
		}
		// Increase descriptors counter
		descCount += imgDescriptors.rows;
		// Increase images counter
		imgIdx++;
	}

	DynamicMat data(descriptorsIndices, keysFilenames, descCount,
			imgDescriptors.cols, imgDescriptors.type());
	/////////////////////////////////////////////////////////////////////

	EXPECT_TRUE(data.rows == imgDescriptors.rows * 2);
	EXPECT_TRUE(data.cols == imgDescriptors.cols);
	EXPECT_TRUE(data.type() == imgDescriptors.type());
	EXPECT_TRUE(
			data.getDescriptorsIndices().size() == descriptorsIndices.size());
	EXPECT_TRUE(data.getKeysFilenames().size() == keysFilenames.size());

}

TEST(DynamicMat, InitByCopy) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;
	std::vector<cv::KeyPoint> imgKeypoints;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		imgDescriptors = cv::Mat();
		imgKeypoints.clear();
		// Load keypoints and descriptors
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);
		// Check that keypoints and descriptors have same length
		for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
			image img;
			img.imgIdx = imgIdx;
			img.startIdx = descCount;
			descriptorsIndices.push_back(img);
		}
		// Increase descriptors counter
		descCount += imgDescriptors.rows;
		// Increase images counter
		imgIdx++;
	}

	DynamicMat data(descriptorsIndices, keysFilenames, descCount,
			imgDescriptors.cols, imgDescriptors.type());
	/////////////////////////////////////////////////////////////////////

	DynamicMat dataCopy(data);

	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());
	EXPECT_TRUE(
			data.getDescriptorsIndices().size()
					== dataCopy.getDescriptorsIndices().size());
	EXPECT_TRUE(
			data.getKeysFilenames().size()
					== dataCopy.getKeysFilenames().size());

}

TEST(DynamicMat, InitByAssignment) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;
	std::vector<cv::KeyPoint> imgKeypoints;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		imgDescriptors = cv::Mat();
		imgKeypoints.clear();
		// Load keypoints and descriptors
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);
		// Check that keypoints and descriptors have same length
		for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
			image img;
			img.imgIdx = imgIdx;
			img.startIdx = descCount;
			descriptorsIndices.push_back(img);
		}
		// Increase descriptors counter
		descCount += imgDescriptors.rows;
		// Increase images counter
		imgIdx++;
	}

	DynamicMat data(descriptorsIndices, keysFilenames, descCount,
			imgDescriptors.cols, imgDescriptors.type());
	/////////////////////////////////////////////////////////////////////

	DynamicMat dataCopy = data;

	EXPECT_TRUE(data.rows == dataCopy.rows);
	EXPECT_TRUE(data.cols == dataCopy.cols);
	EXPECT_TRUE(data.type() == dataCopy.type());
	EXPECT_TRUE(
			data.getDescriptorsIndices().size()
					== dataCopy.getDescriptorsIndices().size());
	EXPECT_TRUE(
			data.getKeysFilenames().size()
					== dataCopy.getKeysFilenames().size());

}

TEST(DynamicMat, RowExtraction) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;
	std::vector<cv::KeyPoint> imgKeypoints;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");
	keysFilenames.push_back("all_souls_000000.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		imgDescriptors = cv::Mat();
		imgKeypoints.clear();
		// Load keypoints and descriptors
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);
		// Check that keypoints and descriptors have same length
		for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
			image img;
			img.imgIdx = imgIdx;
			img.startIdx = descCount;
			descriptorsIndices.push_back(img);
		}
		// Increase descriptors counter
		descCount += imgDescriptors.rows;
		// Increase images counter
		imgIdx++;
	}

	DynamicMat data(descriptorsIndices, keysFilenames, descCount,
			imgDescriptors.cols, imgDescriptors.type());
	/////////////////////////////////////////////////////////////////////

	cv::Mat rowA, rowB;

	for (size_t i = 0; (int) i < data.rows; i++) {
		data.row(i).copyTo(rowA);
		rowB = imgDescriptors.row(i % imgDescriptors.rows);

		// Check number of rows and columns and type to be equal
		EXPECT_TRUE(rowA.rows == 1);
		EXPECT_TRUE(rowA.rows == rowB.rows);
		EXPECT_TRUE(rowA.cols == rowB.cols);
		EXPECT_TRUE(rowA.type() == rowB.type());

		for (size_t j = 0; (int) j < rowA.cols; j++) {
			EXPECT_TRUE(rowA.at<float>(0, j) == rowB.at<float>(0, j));
		}

		rowA.release();
		rowB.release();
	}

}
