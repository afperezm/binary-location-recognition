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

	cv::Mat descriptors(3, 1, CV_8U);
	descriptors.at<uchar>(0, 0) = 4;
	descriptors.at<uchar>(0, 1) = 10;
	descriptors.at<uchar>(0, 2) = 5;
	FileUtils::saveDescriptors("descriptors.bin", descriptors);

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("descriptors.bin");
	vlr::Mat data(descriptorsFilenames);

	vlr::IncrementalKMeans vocabTrainer(data);

	EXPECT_TRUE(vocabTrainer.getDim() == data.cols);

	EXPECT_TRUE(vocabTrainer.getNumDatapoints() == data.rows);

	EXPECT_TRUE(vocabTrainer.getDataset().rows == data.rows);
	EXPECT_TRUE(vocabTrainer.getDataset().cols == data.cols);

	EXPECT_TRUE(vocabTrainer.getCentroids().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getCentroids().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersVariances().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getClustersVariances().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersWeights().rows == 1);
	EXPECT_TRUE(vocabTrainer.getClustersWeights().cols == vocabTrainer.getNumClusters());

	EXPECT_TRUE(vocabTrainer.getClustersSums().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getClustersSums().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersCounts().rows == 1);
	EXPECT_TRUE(vocabTrainer.getClustersCounts().cols == vocabTrainer.getNumClusters());

	EXPECT_TRUE(vocabTrainer.getMiu().rows == 1);
	EXPECT_TRUE(vocabTrainer.getMiu().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getSigma().rows == 1);
	EXPECT_TRUE(vocabTrainer.getSigma().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 0) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 1) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 2) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 3) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 4) == ((double) 1/3));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 5) == ((double) 2/3));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 6) == ((double) 1/3));
	EXPECT_TRUE(vocabTrainer.getMiu().at<double>(0, 7) == ((double) 1/3));

	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 0) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 1) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 2) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 3) == ((double) 0.0));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 4) == sqrt((double) 6/27));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 5) == sqrt((double) 6/27));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 6) == sqrt((double) 6/27));
	EXPECT_TRUE(vocabTrainer.getSigma().at<double>(0, 7) == sqrt((double) 6/27));

	EXPECT_TRUE(vocabTrainer.getNumClusters() >= 0 && vocabTrainer.getOutliers().size() == (size_t) vocabTrainer.getNumClusters());

}

TEST(IncrementalKMeans, InitWithRealData) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeans vocabTrainer(data);

	EXPECT_TRUE(vocabTrainer.getDim() == data.cols);

	EXPECT_TRUE(vocabTrainer.getNumDatapoints() == data.rows);

	EXPECT_TRUE(vocabTrainer.getDataset().rows == data.rows);
	EXPECT_TRUE(vocabTrainer.getDataset().cols == data.cols);

	EXPECT_TRUE(vocabTrainer.getCentroids().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getCentroids().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersVariances().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getClustersVariances().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersWeights().rows == 1);
	EXPECT_TRUE(vocabTrainer.getClustersWeights().cols == vocabTrainer.getNumClusters());

	EXPECT_TRUE(vocabTrainer.getClustersSums().rows == vocabTrainer.getNumClusters());
	EXPECT_TRUE(vocabTrainer.getClustersSums().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getClustersCounts().rows == 1);
	EXPECT_TRUE(vocabTrainer.getClustersCounts().cols == vocabTrainer.getNumClusters());

	EXPECT_TRUE(vocabTrainer.getMiu().rows == 1);
	EXPECT_TRUE(vocabTrainer.getMiu().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getSigma().rows == 1);
	EXPECT_TRUE(vocabTrainer.getSigma().cols == data.cols * 8);

	EXPECT_TRUE(vocabTrainer.getNumClusters() >= 0 && vocabTrainer.getOutliers().size() == (size_t) vocabTrainer.getNumClusters());

}

//	inline class IncrementalKMeansPublic: public vlr::IncrementalKMeans {
//	public:
//		IncrementalKMeansPublic(vlr::IncrementalKMeans vocabTrainer) {
//		}
//	};

TEST(IncrementalKMeans, InitCentroids) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	vocabTrainer.initCentroids();

	cv::Mat min = (vocabTrainer.getMiu() - vocabTrainer.getSigma() / (vocabTrainer.getDim() * 8));
	cv::Mat max = (vocabTrainer.getMiu() + vocabTrainer.getSigma() / (vocabTrainer.getDim() * 8));

	for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
		for (int l = 0; l < vocabTrainer.getCentroids().cols; l++) {
			EXPECT_TRUE(vocabTrainer.getCentroids().at<double>(j, l) >= min.at<double>(0, l));
			EXPECT_TRUE(vocabTrainer.getCentroids().at<double>(j, l) <= max.at<double>(0, l));
		}
	}

}

TEST(IncrementalKMeans, PreComputeDistances) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	vocabTrainer.initCentroids();
	vocabTrainer.preComputeDistances();

	cv::Mat temp;
	for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
		cv::pow(-vocabTrainer.getCentroids().row(j), 2, temp);
		EXPECT_TRUE(vocabTrainer.getClusterDistancesToNullTransaction().at<double>(0, j) == cv::sum(temp).val[0]);
	}

}

TEST(IncrementalKMeans, FindNearestNeighbor) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	vocabTrainer.initCentroids();
	vocabTrainer.preComputeDistances();
	vocabTrainer.initClustersCounters();

	uchar byte = 0;
	cv::Mat transaction;
	int clusterIndex, sparseClusterIndex;
	double distanceToCluster, tempDistanceToCluster, sparseDistanceToCluster;
	for (int i = 0; i < vocabTrainer.getNumDatapoints(); ++i) {
		transaction = data.row(i);
		// Find nearest neighbor to ith transaction
		clusterIndex = 0;
		distanceToCluster = std::numeric_limits<double>::max();
		for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
			tempDistanceToCluster = 0;
			for (int l = 0; l < vocabTrainer.getDim() * 8; l++) {
				if ((l % 8) == 0) {
					byte = *(transaction.col((int) l / 8).data);
				}
				int bit = ((int) ((byte >> (7 - (l % 8))) % 2));
				tempDistanceToCluster += pow(bit - vocabTrainer.getCentroids().at<double>(j, l), 2);
			}
			if (tempDistanceToCluster < distanceToCluster) {
				clusterIndex = j;
				distanceToCluster = tempDistanceToCluster;
			}
		}
		// Find nearest neighbor to ith transaction in a sparse manner
		vocabTrainer.findNearestNeighbor(transaction, sparseClusterIndex, sparseDistanceToCluster);
		EXPECT_TRUE(clusterIndex == sparseClusterIndex);
		EXPECT_TRUE(fabs(distanceToCluster - sparseDistanceToCluster) < 0.00000001);
	}

}

TEST(IncrementalKMeans, SparseSum) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	cv::Mat transaction;
	for (int i = 0; i < vocabTrainer.getNumDatapoints(); ++i) {
		transaction = data.row(i);
		vocabTrainer.initClustersCounters();
		vocabTrainer.sparseSum(transaction, 0);
		for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
			if (j == 0) {
				EXPECT_FALSE(cv::countNonZero(vocabTrainer.getClustersSums().row(j)) == 0);
			} else {
				EXPECT_TRUE(cv::countNonZero(vocabTrainer.getClustersSums().row(j)) == 0);
			}
		}
		int countNonZero = 0;
		for (int l = 0; l < transaction.cols; ++l) {
			countNonZero += FunctionUtils::NumberOfSetBits(transaction.at<uchar>(0, l));
		}
		EXPECT_TRUE(countNonZero == cv::countNonZero(vocabTrainer.getClustersSums().row(0)));
	}

}

TEST(IncrementalKMeans, SparseSubtraction) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	cv::Mat transaction;
	for (int i = 0; i < vocabTrainer.getNumDatapoints(); ++i) {
		transaction = data.row(i);
		vocabTrainer.initClustersCounters();
		vocabTrainer.sparseSubtraction(transaction, 0);
		for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
			if (j == 0) {
				EXPECT_FALSE(cv::countNonZero(vocabTrainer.getClustersSums().row(j)) == 0);
			} else {
				EXPECT_TRUE(cv::countNonZero(vocabTrainer.getClustersSums().row(j)) == 0);
			}
		}
		int countNonZero = 0;
		for (int l = 0; l < transaction.cols; ++l) {
			countNonZero += FunctionUtils::NumberOfSetBits(transaction.at<uchar>(0, l));
		}
		EXPECT_TRUE(countNonZero == cv::countNonZero(vocabTrainer.getClustersSums().row(0)));
	}

}

TEST(IncrementalKMeans, InsertOutlier) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	for (int i = 0; i < 1000; ++i) {
		int clusterIndex = i % vocabTrainer.getNumClusters();
		double distance = rand() % 1000;
		bool expectedResult = true;
		if (vocabTrainer.getOutliers().at(clusterIndex).size() < 10) {
			expectedResult = true;
		} else {
			expectedResult = distance >= vocabTrainer.getOutliers().at(clusterIndex).front().second;
		}
		bool actualResult = vocabTrainer.insertOutlier(i, clusterIndex, distance);
		EXPECT_TRUE(actualResult == expectedResult);
	}

	EXPECT_TRUE(vocabTrainer.getNumClusters() >= 0 && vocabTrainer.getOutliers().size() == (size_t) vocabTrainer.getNumClusters());

	size_t minSize = 10;
	for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
		EXPECT_TRUE(vocabTrainer.getOutliers().at(j).size() >= minSize);
		for (size_t k = 1; k < vocabTrainer.getOutliers().at(j).size(); ++k) {
			EXPECT_TRUE(vocabTrainer.getOutliers().at(j).at(k-1).second >= vocabTrainer.getOutliers().at(j).at(k).second);
		}
	}

}

TEST(IncrementalKMeans, ComputeCentroids) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 3;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	vocabTrainer.initCentroids();
	vocabTrainer.preComputeDistances();
	vocabTrainer.initClustersCounters();

	cv::Mat transaction;
	for (int i = 0; i < vocabTrainer.getNumDatapoints(); ++i) {
		transaction = data.row(i);
		if (i >= 0 && i < vocabTrainer.getNumDatapoints() / 3) {
			vocabTrainer.sparseSum(transaction, 0);
			vocabTrainer.getClustersCounts().col(0) += 1;
		} else if (i >= vocabTrainer.getNumDatapoints() / 3 && i < 2 * vocabTrainer.getNumDatapoints() / 3) {
			vocabTrainer.sparseSum(transaction, 1);
			vocabTrainer.getClustersCounts().col(1) += 1;
		} else {
			vocabTrainer.sparseSum(transaction, 2);
			vocabTrainer.getClustersCounts().col(2) += 1;
		}
	}

	vocabTrainer.computeCentroids(vocabTrainer.getNumDatapoints());

	cv::Mat row1, row2 , result;
	for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
		row1 = vocabTrainer.getCentroids().row(j);
		vocabTrainer.getClustersSums().row(j).convertTo(row2, cv::DataType<double>::type);
		row2 = row2 / ((double) vocabTrainer.getClustersCounts().at<int>(0, j));
		cv::compare(row1, row2, result, cv::CMP_EQ);
		EXPECT_TRUE(cv::sum(result).val[0] == result.cols * 255);
		EXPECT_TRUE(vocabTrainer.getClustersWeights().at<double>(0, j) == (((double) vocabTrainer.getClustersCounts().at<int>(0, j)) / ((double) vocabTrainer.getNumDatapoints())));
	}

}

TEST(IncrementalKMeans, HandleEmptyClusters) {

	std::vector<std::string> descriptorsFilenames;
	descriptorsFilenames.push_back("brief.bin");
	vlr::Mat data(descriptorsFilenames);
	vlr::IncrementalKMeansParams params;
	params["num.clusters"] = 10;
	vlr::IncrementalKMeans vocabTrainer(data, params);

	vocabTrainer.initClustersCounters();

	for (int i = 5; i < vocabTrainer.getNumDatapoints(); ++i) {
		cv::Mat transaction = data.row(i);
		int clusterIndex = (i % (vocabTrainer.getNumClusters() - 1));
		vocabTrainer.sparseSum(transaction, clusterIndex);
		vocabTrainer.getClustersCounts().col(clusterIndex) += 1;
	}

	for (int j = 0; j < vocabTrainer.getNumClusters(); ++j) {
		vocabTrainer.getClustersWeights().col(j) = ((double) vocabTrainer.getClustersCounts().at<int>(0, j)) / ((double) vocabTrainer.getNumDatapoints());
	}

	vocabTrainer.insertOutlier(0, 0, 0.0);
	vocabTrainer.insertOutlier(1, 0, 1.0);
	vocabTrainer.insertOutlier(2, 0, 10.0);
	vocabTrainer.insertOutlier(3, 0, 3.0);
	vocabTrainer.insertOutlier(4, 0, 4.0);

	EXPECT_TRUE(vocabTrainer.getOutliers().at(0).size() == (size_t) 5);
	EXPECT_TRUE(vocabTrainer.getClustersCounts().at<int>(0, vocabTrainer.getNumClusters() - 1) == 0);
	cv::Mat expected = vocabTrainer.getClustersSums().row(vocabTrainer.getNumClusters() - 1);

	vocabTrainer.handleEmptyClusters();

	EXPECT_TRUE(vocabTrainer.getOutliers().at(0).size() == (size_t) 4);
	EXPECT_TRUE(vocabTrainer.getClustersCounts().at<int>(0, vocabTrainer.getNumClusters() - 1) == 1);
	cv::Mat transaction = data.row(2);
	vocabTrainer.sparseSubtraction(transaction, vocabTrainer.getNumClusters() - 1);
	cv::Mat actual = vocabTrainer.getClustersSums().row(vocabTrainer.getNumClusters() - 1);
	cv::Mat result;
	cv::compare(expected, actual, result, cv::CMP_EQ);
	EXPECT_TRUE(cv::sum(result).val[0] == result.cols * 255);

}

} /* namespace vlr */
