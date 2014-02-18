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

TEST(FileStorageIOReal, LoadSave) {

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

TEST(FileStorageIOReal, LoadStress) {

	cvflann::Logger::setDestination("load_times_(real-valued).log");

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

TEST(FileStorageIOBin, LoadStress) {

	cvflann::Logger::setDestination("load_times_(bin-valued).log");

	cv::Mat descriptors;

	for (size_t i = 0; i < 50; i++) {
		descriptors = cv::Mat();

		double mytime = cv::getTickCount();
		FileUtils::loadDescriptors("brief_0.yaml.gz", descriptors);
		mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency()
				* 1000.0;

		cvflann::Logger::log(0, "%lf\n", mytime);
	}
}

TEST(STLIOReal, LoadSave) {
	cv::Mat original;
	FileUtils::loadDescriptors("sift_0.yaml.gz", original);
	FileUtils::saveDescriptorsToBin("sift_0.bin", original);

	cv::Mat loaded;
	FileUtils::loadDescriptorsFromBin("sift_0.bin", loaded);

	// Check number of rows, columns and type are equal
	EXPECT_TRUE(original.rows == loaded.rows);
	EXPECT_TRUE(original.cols == loaded.cols);
	EXPECT_TRUE(original.type() == loaded.type());

	// Check elements are equal
	for (int i = 0; i < original.rows; ++i) {
		for (int j = 0; j < original.cols; ++j) {
			EXPECT_TRUE(original.at<float>(i, j) == loaded.at<float>(i, j));
		}
	}

//	delete[] loaded.data;

}

TEST(STLIOBin, LoadSave) {
	cv::Mat original;
	FileUtils::loadDescriptors("brief_0.yaml.gz", original);
	FileUtils::saveDescriptorsToBin("brief_0.bin", original);

	cv::Mat loaded;
	FileUtils::loadDescriptorsFromBin("brief_0.bin", loaded);

	// Check number of rows, columns and type are equal
	EXPECT_TRUE(original.rows == loaded.rows);
	EXPECT_TRUE(original.cols == loaded.cols);
	EXPECT_TRUE(original.type() == loaded.type());

	// Check elements are equal
	for (int i = 0; i < original.rows; ++i) {
		for (int j = 0; j < original.cols; ++j) {
			EXPECT_TRUE(original.at<uchar>(i, j) == loaded.at<uchar>(i, j));
		}
	}

//	delete[] loaded.data;
}

TEST(STLIOReal, LoadSaveRow) {

	cv::Mat original, row;

	FileUtils::loadDescriptors("sift_0.yaml.gz", original);
	FileUtils::saveDescriptorsToBin("sift_0.bin", original);

	for (int rowIdx = 0; rowIdx < original.rows; ++rowIdx) {

		FileUtils::loadDescriptorsRowFromBin("sift_0.bin", row, rowIdx);

		// Check number of rows, columns and type are equal
		CV_Assert(original.cols == row.cols);
		CV_Assert(original.type() == row.type());

		// Check elements are equal
		for (int j = 0; j < original.cols; ++j) {
//			printf("(%d, %d) %f %f\n", rowIdx, j,
//					original.row(rowIdx).at<float>(0, j), row.at<float>(0, j));
			CV_Assert(
					original.row(rowIdx).at<float>(0, j)
							== row.at<float>(0, j));
		}

		delete[] row.data;

	}

}

TEST(STLIOBin, LoadSaveRow) {

	cv::Mat original, row;

	FileUtils::loadDescriptors("brief_0.yaml.gz", original);
	FileUtils::saveDescriptorsToBin("brief_0.bin", original);

	for (int rowIdx = 0; rowIdx < original.rows; ++rowIdx) {

		FileUtils::loadDescriptorsRowFromBin("brief_0.bin", row, rowIdx);

		// Check number of rows, columns and type are equal
		CV_Assert(original.cols == row.cols);
		CV_Assert(original.type() == row.type());

		// Check elements are equal
		for (int j = 0; j < original.cols; ++j) {
//			printf("(%d, %d) %f %f\n", rowIdx, j,
//					original.row(rowIdx).at<float>(0, j), row.at<float>(0, j));
			CV_Assert(
					original.row(rowIdx).at<unsigned char>(0, j)
							== row.at<unsigned char>(0, j));
		}

		delete[] row.data;

	}

}
