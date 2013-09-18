/*
 * clustering.hpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#ifndef CLUSTERING_HPP_
#define CLUSTERING_HPP_

#include <kmajority_index.h>

namespace clustering {

/**
 * Performs kmeans clustering of the points passed as argument using
 * Hamming distance and a majority voting scheme for centers computation.
 *
 * @param data - Matrix containing data for clustering.
 * @param K - Number of clusters to split the set by.
 * @param labels - Output matrix that stores the cluster indices for every sample.
 * @param criteria - The algorithm termination criteria, that is, the maximum number of iterations and/or the desired accuracy.
 * @param flags - Flag indicating initialization method, one of KMEANS_RANDOM_CENTERS, KMEANS_PP_CENTERS or KMEANS_USE_INITIAL_LABELS
 * @param centers - Output matrix of the cluster centers, one row per each cluster center.
 *
 * @return The compactness measure of the resulting clustering computed
 *         as the sum of squared distances from each transaction to its center.
 */
void kmajority(const cv::Mat data, const int K, cv::Mat labels,
		const cv::TermCriteria criteria, const int flags, cv::Mat centers) {

//	cvflann::KMeansIndexParams params;
//	int branching_ = cvflann::get_param(params, "branching", 32);
//	int iterations_ = cvflann::get_param(params, "iterations", 11);

	cv::Ptr<KMajorityIndex> kMajIdx = new KMajorityIndex(K, criteria.maxCount);
	kMajIdx->cluster(data);

	centers = kMajIdx->getCentroids();
	kMajIdx->getLabels(labels);

	return;
}

}
// namespace name

#endif /* CLUSTERING_HPP_ */
