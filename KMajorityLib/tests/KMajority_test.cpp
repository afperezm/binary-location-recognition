/*
 * KMajority_test.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <ctime>

#include <gtest/gtest.h>

#include <Clustering.h>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <KMajority.h>

TEST(KMajority, InstantiateOnHeap) {
	cv::Ptr<KMajority> obj;

	EXPECT_TRUE(obj == NULL);

	vlr::Mat emptyMat = vlr::Mat();

	obj = new KMajority(0, 0, emptyMat);

	EXPECT_TRUE(obj != NULL);
}

TEST(KMajority, InstantiateOnStack) {

	vlr::Mat emptyMat = vlr::Mat();

	KMajority obj(0, 0, emptyMat);

	EXPECT_TRUE(obj.getCentroids().empty());

}

TEST(KMajority, CumBitSum) {

}

TEST(KMajority, MajorityVoting) {

}

TEST(KMajority, Clustering) {

	std::vector<std::string> filenames;
	filenames.push_back("brief_0.yaml.gz");

	vlr::Mat descriptors(filenames);

	KMajority bofModel(10, 10, descriptors, vlr::indexType::HIERARCHICAL);

	bofModel.cluster();

	EXPECT_FALSE(bofModel.getCentroids().empty());

	EXPECT_TRUE(bofModel.getCentroids().rows == 10);

	EXPECT_TRUE(
			descriptors.rows >= 0
					&& (size_t ) descriptors.rows
							== bofModel.getClusterAssignments().size());

	// Check all data has been assigned to same cluster
	int cumRes = 0;
	for (int k = 0; k < int(bofModel.getClusterCounts().size()); ++k) {
		cumRes = cumRes + bofModel.getClusterCounts().at(k);
	}

	EXPECT_TRUE(cumRes == descriptors.rows);

}

TEST(KMajority, SaveLoad) {

	std::vector<std::string> filenames;
	filenames.push_back("brief_0.yaml.gz");
	vlr::Mat descriptors(filenames);

	KMajority bofModel(10, 10, descriptors, vlr::indexType::LINEAR);
	bofModel.cluster();
	bofModel.save("test_vocab.yaml.gz");

	KMajority bofModelLoaded;
	bofModelLoaded.load("test_vocab.yaml.gz");

	EXPECT_TRUE(
			std::equal(bofModel.getCentroids().begin<uchar>(),
					bofModel.getCentroids().end<uchar>(),
					bofModelLoaded.getCentroids().begin<uchar>()));

}

TEST(KMajority, Regression) {

	std::vector<std::string> filenames;
	filenames.push_back("brief_0.yaml.gz");

	vlr::Mat descriptors(filenames);

	cv::Mat centroids;
	std::vector<int> labels;

	double mytime = cv::getTickCount();

	clustering::kmajority(16, 100, descriptors, centroids, labels);

	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("Clustered [%d] points into [%d] clusters in [%lf] ms\n",
			descriptors.rows, centroids.rows, mytime);

	for (int j = 0; j < centroids.rows; ++j) {
		printf("   Cluster %d:\n", j + 1);
		FunctionUtils::printDescriptors(centroids.row(j));
	}

	for (int i = 0; i < descriptors.rows; ++i) {
		printf("   Data point [%d] was assigned to cluster [%d]\n", i,
				labels[i]);
	}

	EXPECT_FALSE(centroids.empty());
	EXPECT_FALSE(labels.empty());
}
