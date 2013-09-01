/*
 * KMajority.h
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#ifndef KMAJORITY_H_
#define KMAJORITY_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

class KMajority {
public:

	KMajority(unsigned int _k, unsigned int _max_iterations) :
			k(_k), max_iterations(_max_iterations), dim(-1), n(-1), belongs_to(
					new unsigned int[0]), distance_to(new unsigned int[0]) {
		;
	}

	~KMajority() {
		delete[] belongs_to;
		delete[] distance_to;
	}

	/**
	 * Implements majority voting scheme as explained in Grana2013 for centroid computation
	 * based on component wise majority of bits from descriptors matrix.
	 *
	 * @param descriptors binary matrix of size (n x d)
	 * @param centroids matrix where centroids will be saved
	 */
	void computeCentroid(const cv::Mat& descriptors, cv::Mat centroid) const;

	void computeCentroids(const std::vector<cv::KeyPoint>& keypoints,
			const cv::Mat& descriptors);

	/**
	 * Computes Hamming distance between each descriptor and each cluster centroid
	 * and assigns a label to the class_id attribute of the keypoints vector.
	 * @param keypoints
	 * @param descriptors binary matrix of size (n x d)
	 * @param centroids binary matrix of size (1 x d)
	 */
	bool quantize(std::vector<cv::KeyPoint>& keypoints,
			const cv::Mat& descriptors);

	void cluster(std::vector<cv::KeyPoint>& keypoints,
			const cv::Mat& descriptors);

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
	// Matrix of centroids
	cv::Mat centroids;

	void initCentroids(const cv::Mat& descriptors);
};

#endif /* KMAJORITY_H_ */
