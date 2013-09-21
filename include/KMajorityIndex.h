/*
 * kmajority_index.h
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#ifndef KMAJORITY_INDEX_H_
#define KMAJORITY_INDEX_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class KMajorityIndex {
public:

	/**
	 *
	 * @param _k
	 * @param _max_iterations
	 * @param _data
	 * @param _indices
	 * @param _indices_length
	 */
	KMajorityIndex(unsigned int _k, unsigned int _max_iterations,
			const cv::Mat& _data, cv::Ptr<int>& _indices,
			const int& _indices_length);

	~KMajorityIndex();

	/**
	 * Implements k-means clustering loop.
	 *
	 * @param indices - The set of indices indicating which data points should be clustered
	 */
	void cluster();

	const cv::Mat& getCentroids() const;

	uint* getClusterCounts() const;

	uint* getClusterAssignments() const;

	int getNumberOfClusters() const;

private:
	// Number of clusters
	uint k;
	// Maximum number of iterations
	uint max_iterations;
	// Reference to the matrix with data to cluster
	const cv::Mat& data;
	// Array of indices indicating data points involved in the clustering process
	cv::Ptr<int>& indices;
	// Number of indices
	const int& indices_length;
	// Dimensionality (in Bytes)
	uint dim;
	// Number of data instances
	uint n;
	// List of the cluster each data point belongs to
	uint* belongs_to;
	// List of distance from each data point to the cluster it belongs to
	uint* distance_to;
	// Number of transactions assigned to each cluster
	uint* cluster_counts;
	// Matrix of clusters centers
	cv::Mat centroids;

	/**
	 * Initializes cluster centers choosing among the data points indicated by indices.
	 *
	 * @param indices - The set of indices among which to choose
	 */
	void initCentroids();

	/**
	 * Implements majority voting scheme for cluster centers computation
	 * based on component wise majority of bits from data matrix
	 * as proposed by Grana2013.
	 *
	 * @param indices - The set of indices indicating the data points
	 * 					involved in the cluster centers computation
	 */
	void computeCentroids();

	/**
	 * Computes Hamming distance between each descriptor and each cluster center.
	 *
	 * @param indices - The set of indices indicating the data points to quantize
	 */
	bool quantize();

};

#endif /* KMAJORITY_INDEX_H_ */
