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
#include <KMajorityIndex.h>

TEST(KMajority, InstantiateOnHeap) {
	cv::Ptr<KMajority> obj;

	EXPECT_TRUE(obj == NULL);

	obj = new KMajority(0, 0, cv::Mat());

	EXPECT_TRUE(obj != NULL);
}

TEST(KMajority, InstantiateOnStack) {

	KMajority obj(0, 0, cv::Mat());

	EXPECT_TRUE(obj.getCentroids().empty());

}

TEST(KMajority, CumBitSum) {

}

TEST(KMajority, MajorityVoting) {

}

TEST(KMajority, Clustering) {

	cv::Mat descriptors;
	FileUtils::loadDescriptors("brief_0.yaml.gz", descriptors);

	KMajority bofModel(10, 10, descriptors, vlr::indexType::HIERARCHICAL);

	bofModel.cluster();

	for (int k = 0; k < int(bofModel.getClusterCounts().size()); ++k) {
		printf("%d) %d\n", k, bofModel.getClusterCounts().at(k));
	}

}

TEST(KMajority, Regression) {

	cv::Mat descriptors;
	FileUtils::loadDescriptors("brief_0.yaml.gz", descriptors);

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
