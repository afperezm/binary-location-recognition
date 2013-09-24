/*
 * Clustering.hpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#ifndef CLUSTERING_HPP_
#define CLUSTERING_HPP_

#include <KMajorityIndex.h>

namespace clustering {

/**
 * Performs k-means clustering of the data using Hamming distance and
 * a majority voting scheme for centers computation.
 *
 * @param k - Number of clusters
 * @param max_iterations - Maximum number of iterations
 * @param data - Data to cluster composed of n d-dimensional features
 * @param labels - Reference to an integer array with the assignment of data to clusters
 * @param centroids - Reference to a matrix of k d-dimensional centroids
 */
void kmajority(int k, int max_iterations, const cv::Mat& data,
		cv::Ptr<int>& indices, const int& indices_length, cv::Mat& labels,
		cv::Mat& centroids) {

	cv::Ptr<KMajorityIndex> kMajIdx = new KMajorityIndex(k, max_iterations,
			data, indices, indices_length);

	kMajIdx->cluster();

	centroids = kMajIdx->getCentroids();

	labels = cv::Mat();
	labels.create(indices_length, 1, cv::DataType<uint>::type);
	for (size_t i = 0; (int) i < indices_length; i++) {
		labels.at<uint>(i, 1) = kMajIdx->getClusterAssignments()[i];
	}
}

}
// namespace name

#endif /* CLUSTERING_HPP_ */
