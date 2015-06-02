/*
 * IncrementalKMeans_test.cpp
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#include <gtest/gtest.h>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <IncrementalKMeans.hpp>

namespace vlr {

TEST(IncrementalKMeans, InitWithFakeData) {

	cv::Mat data(3, 1, CV_8U);

	data.at<uchar>(0, 0) = 4;
	data.at<uchar>(0, 1) = 10;
	data.at<uchar>(0, 2) = 5;

	vlr::IncrementalKMeans obj(data);

	EXPECT_TRUE(obj.getDim() == data.cols);

	EXPECT_TRUE(obj.getNumDatapoints() == data.rows);

	EXPECT_TRUE(obj.getDataset().rows == data.rows);
	EXPECT_TRUE(obj.getDataset().cols == data.cols);

	EXPECT_TRUE(obj.getCentroids().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getCentroids().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersVariances().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getClustersVariances().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersWeights().rows == 1);
	EXPECT_TRUE(obj.getClustersWeights().cols == obj.getNumClusters());

	EXPECT_TRUE(obj.getClustersSums().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getClustersSums().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersCounts().rows == 1);
	EXPECT_TRUE(obj.getClustersCounts().cols == obj.getNumClusters());

	EXPECT_TRUE(obj.getMiu().rows == 1);
	EXPECT_TRUE(obj.getMiu().cols == data.cols * 8);

	EXPECT_TRUE(obj.getSigma().rows == 1);
	EXPECT_TRUE(obj.getSigma().cols == data.cols * 8);

	EXPECT_TRUE(obj.getMiu().at<double>(0, 0) == ((double) 0.0));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 1) == ((double) 0.0));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 2) == ((double) 0.0));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 3) == ((double) 0.0));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 4) == ((double) 1/3));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 5) == ((double) 2/3));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 6) == ((double) 1/3));
	EXPECT_TRUE(obj.getMiu().at<double>(0, 7) == ((double) 1/3));

	EXPECT_TRUE(obj.getSigma().at<double>(0, 0) == ((double) 0.0));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 1) == ((double) 0.0));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 2) == ((double) 0.0));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 3) == ((double) 0.0));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 4) == sqrt((double) 6/27));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 5) == sqrt((double) 6/27));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 6) == sqrt((double) 6/27));
	EXPECT_TRUE(obj.getSigma().at<double>(0, 7) == sqrt((double) 6/27));

	EXPECT_TRUE(obj.getOutliers().empty());

}

TEST(IncrementalKMeans, InitWithRealData) {

	cv::Mat data;
	FileUtils::loadDescriptors("brief_0.bin", data);

	vlr::IncrementalKMeans obj(data);

	EXPECT_TRUE(obj.getDim() == data.cols);

	EXPECT_TRUE(obj.getNumDatapoints() == data.rows);

	EXPECT_TRUE(obj.getDataset().rows == data.rows);
	EXPECT_TRUE(obj.getDataset().cols == data.cols);

	EXPECT_TRUE(obj.getCentroids().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getCentroids().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersVariances().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getClustersVariances().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersWeights().rows == 1);
	EXPECT_TRUE(obj.getClustersWeights().cols == obj.getNumClusters());

	EXPECT_TRUE(obj.getClustersSums().rows == obj.getNumClusters());
	EXPECT_TRUE(obj.getClustersSums().cols == data.cols * 8);

	EXPECT_TRUE(obj.getClustersCounts().rows == 1);
	EXPECT_TRUE(obj.getClustersCounts().cols == obj.getNumClusters());

	EXPECT_TRUE(obj.getMiu().rows == 1);
	EXPECT_TRUE(obj.getMiu().cols == data.cols * 8);

	EXPECT_TRUE(obj.getSigma().rows == 1);
	EXPECT_TRUE(obj.getSigma().cols == data.cols * 8);

	EXPECT_TRUE(obj.getOutliers().empty());

}

//TEST(IncrementalKMeans, Clustering) {
//
//	cv::Mat descriptors;
//	FileUtils::loadDescriptors("brief_0.bin", descriptors);
//
//	vlr::IncrementalKMeans obj(descriptors);
//
//	obj.build();
//
//}

} /* namespace vlr */
