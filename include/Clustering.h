/*
 * clustering.hpp
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
		cv::Ptr<int>& _indices, const int& _indices_length,
		cv::Ptr<uint>& labels, cv::Mat& centroids) {

	cv::Ptr<KMajorityIndex> kMajIdx = new KMajorityIndex(k, max_iterations,
			data, _indices, _indices_length);

	kMajIdx->cluster();

	centroids = kMajIdx->getCentroids();
	labels = kMajIdx->getClusterAssignments();
}

}
// namespace name

#endif /* CLUSTERING_HPP_ */
