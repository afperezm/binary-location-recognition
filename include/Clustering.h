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
 * @param data - data to cluster composed of n d-dimensional features
 * @param labels - Matrix with the assignment of data to clusters
 * @param centroids - Matrix of k d-dimensional centroids
 */
void kmajority(int k, int max_iterations, const cv::Mat& data, cv::Mat& labels,
		cv::Mat& centroids) {

	cv::Ptr<KMajorityIndex> kMajIdx = new KMajorityIndex(k, max_iterations);
	kMajIdx->cluster(data);

	centroids = kMajIdx->getCentroids();
	kMajIdx->getLabels(labels);
}

}
// namespace name

#endif /* CLUSTERING_HPP_ */
