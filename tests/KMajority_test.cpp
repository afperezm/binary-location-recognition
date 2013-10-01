/*
 * KMajority_test.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <KMajorityIndex.h>

// Step 5a/5: Cluster descriptors using k-majority

//	cv::Ptr<int> indices = new int[descriptors.rows];
//	for (size_t i = 0; i < (size_t) descriptors.rows; ++i) {
//		indices[i] = int(i);
//	}
//	uint* labels = new uint[0];
//	cv::Mat centroids;
//	mytime = cv::getTickCount();
//	std::srand(unsigned(std::time(0)));
//	clustering::kmajority(16, 100, descriptors, indices, descriptors.rows,
//			centroids, labels);
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;
//	printf("-- Clustered [%zu] keypoints in [%d] clusters in [%lf] ms\n",
//			keypoints_1.size(), centroids.rows, mytime);
//
//	for (size_t j = 0; (int) j < centroids.rows; j++) {
//		//printf("   Cluster %u has %u transactions assigned\n", j + 1, kMajIdx->getClusterCounts()[j]);
//		printf("   Cluster %lu:\n", j + 1);
//		printDescriptors(centroids.row(j));
//	}
