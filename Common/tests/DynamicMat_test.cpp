/*
 * FunctionUtils.cpp
 *
 *  Created on: Nov 3, 2013
 *      Author: andresf
 */

#include <vector>
#include <string>
#include <gtest/gtest.h>
#include <FunctionUtils.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <FileUtils.hpp>

TEST(DynamicMat, EmptyInstantiation) {

	DynamicMat data;

	EXPECT_TRUE(data.rows == 0);
	EXPECT_TRUE(data.cols == 0);
	EXPECT_TRUE(data.type() == -1);
	EXPECT_TRUE(data.getDescriptorsIndices().size() == 0);
	EXPECT_TRUE(data.getKeysFilenames().size() == 0);

}

TEST(DynamicMat, Instantiation) {

	// Prepare keypoints and descriptors. Recall: 128 dimensions in SIFT
	cv::Mat imgDescriptors(3000, 128, cv::DataType<float>::type);
	cv::randn(imgDescriptors, cv::Scalar(128), cv::Scalar(128));
	std::vector<cv::KeyPoint> imgKeypoints;
	imgKeypoints.reserve(imgDescriptors.rows);
	imgKeypoints.insert(imgKeypoints.begin(), imgDescriptors.rows,
			cv::KeyPoint());
	FileUtils::saveFeatures("sift_descriptors.yaml.gz", imgKeypoints,
			imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_descriptors.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		// Load keypoints and descriptors
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

	EXPECT_TRUE(data.rows == imgDescriptors.rows);
	EXPECT_TRUE(data.cols == imgDescriptors.cols);
	EXPECT_TRUE(data.type() == imgDescriptors.type());
	EXPECT_TRUE(
			data.getDescriptorsIndices().size() == descriptorsIndices.size());
	EXPECT_TRUE(data.getKeysFilenames().size() == keysFilenames.size());

}

TEST(DynamicMat, InitByCopy) {

	// Prepare keypoints and descriptors. Recall: 128 dimensions in SIFT
	cv::Mat imgDescriptors(3000, 128, cv::DataType<float>::type);
	cv::randn(imgDescriptors, cv::Scalar(128), cv::Scalar(128));
	std::vector<cv::KeyPoint> imgKeypoints;
	imgKeypoints.reserve(imgDescriptors.rows);
	imgKeypoints.insert(imgKeypoints.begin(), imgDescriptors.rows,
			cv::KeyPoint());
	FileUtils::saveFeatures("sift_descriptors.yaml.gz", imgKeypoints,
			imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_descriptors.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		// Load keypoints and descriptors
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

	// Prepare keypoints and descriptors. Recall: 128 dimensions in SIFT
	cv::Mat imgDescriptors(3000, 128, cv::DataType<float>::type);
	cv::randn(imgDescriptors, cv::Scalar(128), cv::Scalar(128));
	std::vector<cv::KeyPoint> imgKeypoints;
	imgKeypoints.reserve(imgDescriptors.rows);
	imgKeypoints.insert(imgKeypoints.begin(), imgDescriptors.rows,
			cv::KeyPoint());
	FileUtils::saveFeatures("sift_descriptors.yaml.gz", imgKeypoints,
			imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_descriptors.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		// Load keypoints and descriptors
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

	// Prepare keypoints and descriptors. Recall: 128 dimensions in SIFT
	cv::Mat imgDescriptors(3000, 128, cv::DataType<float>::type);
	cv::randn(imgDescriptors, cv::Scalar(128), cv::Scalar(128));
	std::vector<cv::KeyPoint> imgKeypoints;
	imgKeypoints.reserve(imgDescriptors.rows);
	imgKeypoints.insert(imgKeypoints.begin(), imgDescriptors.rows,
			cv::KeyPoint());
	FileUtils::saveFeatures("sift_descriptors.yaml.gz", imgKeypoints,
			imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_descriptors.yaml.gz");
	keysFilenames.push_back("sift_descriptors.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		// Load keypoints and descriptors
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

	cv::Mat rowA;
	data.row(0).copyTo(rowA);
	cv::Mat rowB = imgDescriptors.row(0);

	EXPECT_TRUE(rowA.rows == 1);
	EXPECT_TRUE(rowB.rows == 1);

	EXPECT_TRUE(rowA.cols == rowB.cols);

	for (size_t i = 0; (int) i < rowA.cols; i++) {
		EXPECT_TRUE(rowA.at<float>(0, i) == rowB.at<float>(0, i));
	}

	data.row(imgDescriptors.rows).copyTo(rowA);

	EXPECT_TRUE(rowA.rows == 1);

	EXPECT_TRUE(rowA.cols == rowB.cols);

	for (size_t i = 0; (int) i < rowA.cols; i++) {
		EXPECT_TRUE(rowA.at<float>(0, i) == rowB.at<float>(0, i));
	}

}
