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

	KMajorityIndex(unsigned int _k, unsigned int _max_iterations) :
			k(_k), max_iterations(_max_iterations), dim(-1), n(-1), belongs_to(
					new unsigned int[0]), distance_to(new unsigned int[0]), cluster_counts(
					new unsigned int[0]) {
		;
	}

	~KMajorityIndex() {
		delete[] belongs_to;
		delete[] distance_to;
		delete[] cluster_counts;
	}

	/**
	 * Implements majority voting scheme as explained in Grana2013 for centroid computation
	 * based on component wise majority of bits from descriptors matrix.
	 *
	 * @param descriptors - Binary matrix of size (n x d)
	 */
	void computeCentroids(const cv::Mat& descriptors);

	/**
	 * Computes Hamming distance between each descriptor and each cluster centroid.
	 *
	 * @param descriptors binary matrix of size (n x d)
	 */
	bool quantize(const cv::Mat& descriptors);

	void cluster(const cv::Mat& descriptors);

	const cv::Mat& getCentroids() const {
		return centroids;
	}

	unsigned int* getClusterCounts() const {
		return cluster_counts;
	}

	unsigned int getNumberOfClusters() const {
		return k;
	}

	/**
	 * Return the cluster indexes each transaction is assigned to.
	 *
	 * @param labels - Output matrix storing the transactions cluster labels
	 */
	void getLabels(cv::Mat& labels);

private:
	// Number of clusters
	unsigned int k;
	// Maximum number of iterations
	unsigned int max_iterations;
	// Dimensionality (in Bytes)
	unsigned int dim;
	// Number of data instances
	unsigned int n;
	// List of the cluster each transaction belongs to
	unsigned int* belongs_to;
	// List of distance from each transaction to the cluster it belongs to
	unsigned int* distance_to;
	// Number of transactions assigned to each cluster
	unsigned int* cluster_counts;
	// Matrix of centroids
	cv::Mat centroids;

	void initCentroids(const cv::Mat& descriptors);
};

#endif /* KMAJORITY_INDEX_H_ */
