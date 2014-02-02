/*
 * Clustering.hpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#ifndef CLUSTERING_HPP_
#define CLUSTERING_HPP_

#include <KMajority.h>

namespace clustering {

/**
 * Performs k-means clustering of the data using Hamming distance and
 * a majority voting scheme for centers computation.
 *
 * @param k - Number of clusters
 * @param maxIterations - Maximum number of iterations
 * @param data - Data to cluster composed of n d-dimensional features
 * @param centers - Reference to a matrix of k d-dimensional centers
 * @param labels
 * @param centersInit
 */
inline void kmajority(int k, int maxIterations, vlr::Mat& data,
		cv::Mat& centers, std::vector<int>& labels,
		cvflann::flann_centers_init_t centersInit =
				cvflann::FLANN_CENTERS_RANDOM) {

	cv::Ptr<vlr::KMajority> bofModel = new vlr::KMajority(k, maxIterations, data,
			vlr::indexType::LINEAR, centersInit);

	bofModel->cluster();

	centers = bofModel->getCentroids();

	labels = bofModel->getClusterAssignments();

}

} /* namespace clustering */

#endif /* CLUSTERING_HPP_ */
