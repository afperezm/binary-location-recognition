/*
 * KMajority.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <KMajority.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

#include <iostream>
#include <functional>
#include <bitset>
#include <ctime>

typedef cvflann::Hamming<uchar> Distance;
typedef typename Distance::ResultType DistanceType;

void KMajority::cluster(std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {

	if (descriptors.type() != CV_8U) {
		throw std::string(
				"KMajority::cluster: error, descriptors matrix is not binary");
	}

	this->n = descriptors.rows;
	this->belongs_to = new unsigned int[this->n];
	std::fill_n(this->belongs_to, this->n, -1);
	this->distance_to = new unsigned int[this->n];
	std::fill_n(this->distance_to, this->n, -1);

	this->dim = descriptors.cols;

	// Randomly generate clusters
	this->initCentroids(descriptors);

	// Assign data to clusters
	this->quantize(keypoints, descriptors);

	bool converged = false;

	int iteration = 0;
	while (converged == false && iteration < max_iterations) {
		iteration++;
		// Compute the new cluster centers
		try {
			this->computeCentroids(keypoints, descriptors);
		} catch (cv::Exception& e) {
			fprintf(stderr, "Error while computing centroids: [%s]",
					e.err.c_str());
		}
		// Reassign data to clusters
		converged = this->quantize(keypoints, descriptors);

		// Compute number of assigned transaction to each cluster
//		unsigned int* cluster_counts = new unsigned int[k];
//		std::fill_n(cluster_counts, k, 0);
//		for (unsigned int i = 0; i < n; i++) {
//			cluster_counts[belongs_to[n]]++;
//		}
		// Find empty clusters
//		for (unsigned int j = 0; j < k; j++) {
//			if (cluster_counts[j] == 0) {
//				// Find farthest element to its assigned cluster
//				int farthest_element_idx = 0;
//				for (unsigned int i = 1; i < n; i++) {
//					if (distance_to[i] > distance_to[farthest_element_idx]) {
//						farthest_element_idx = i;
//					}
//				}
//				// Re-assign farthest_element to the found empty cluster
//				cluster_counts[belongs_to[farthest_element_idx]]--;
//				belongs_to[farthest_element_idx] = j;
//				cluster_counts[j]++;
//				keypoints[farthest_element_idx].class_id = j;
//				distance_to[farthest_element_idx] = hammingDistance(
//						descriptors.row(farthest_element_idx),
//						centroids.row(j));
//			}
//		}
	}

}

void chooseCentersRandom(int k, int* indices, int indices_length, int* centers,
		int& centers_length) {
	cvflann::UniqueRandom r(indices_length);

	int index;
	for (index = 0; index < k; ++index) {
		bool duplicate = true;
		int rnd;
		while (duplicate) {
			duplicate = false;
			rnd = r.next();
			if (rnd < 0) {
				centers_length = index;
				return;
			}

			centers[index] = indices[rnd];

//			for (int j = 0; j < index; ++j) {
//				DistanceType sq = distance_(dataset_[centers[index]],
//						dataset_[centers[j]], dataset_.cols);
//				if (sq < 1e-16) {
//					duplicate = true;
//				}
//			}
		}
	}

	centers_length = index;
}

void KMajority::initCentroids(const cv::Mat& descriptors) {

	cv::Ptr<int> centers_idx = new int[k];
	int centers_length;
	cv::Ptr<int> indices = new int[n];

	for (int i = 0; i < this->n; i++) {
		indices[i] = i;
	}

	std::srand(unsigned(std::time(0)));

	chooseCentersRandom(k, indices, n, centers_idx, centers_length);

	centroids.create(centers_length, dim, descriptors.type());

	for (int i = 0; i < centers_length; i++) {
		descriptors.row(centers_idx[i]).copyTo(
				centroids(cv::Range(i, i + 1), cv::Range(0, this->dim)));
	}

}

bool KMajority::quantize(std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {

	bool converged = true;

	// Comparison of all descriptors vs. all centroids
	for (int i = 0; i < this->n; i++) {
		// Set minimum distance as the distance to its assigned cluster or to the maximum representable integer
		int min_hd = distance_to[i] >= 0 ? distance_to[i] : INT_MAX;
		for (int j = 0; j < this->k; j++) {
			// TODO Check execution time and see if it can be optimized doing bit counts
			// maybe instead of storing the descriptors as a matrix of uchar store a matrix of integers or doubles

			// Compute hamming distance between ith descriptor and jth cluster
			cv::Hamming distance;
			int hd = distance(descriptors.row(i), centroids.row(j), descriptors.cols);

			// Update cluster assignment to the nearest cluster
			if (hd < min_hd) {
				min_hd = hd;
				if (belongs_to[i] != j) {
					converged = false;
				}
				belongs_to[i] = j;
				distance_to[i] = hd;
				keypoints[i].class_id = j;
			}
		}
	}

	return converged;
}

void KMajority::computeCentroids(const std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {
	for (int i = 0; i < k; i++) {
		cv::Mat clusterMask = cv::Mat::zeros(descriptors.size(),
				descriptors.type());
		cv::Mat colwiseCum = cv::Mat::zeros(1, this->dim, CV_32F);
		for (int j = 0; j < n; j++) {
			if (belongs_to[j] == i) {
//				for (int k = 0; k < dim; k++) { }
				descriptors.row(j).copyTo(
						clusterMask(cv::Range(j, j + 1),
								cv::Range(0, clusterMask.cols)));
			}
		}
		// Threshold the resulting cumulative sum vector
		computeCentroid(clusterMask,
				centroids(cv::Range(i, i + 1), cv::Range(0, centroids.cols)));
	}
}

void KMajority::computeCentroid(const cv::Mat& descriptors,
		cv::Mat centroid) const {

// Initialize matrices
	cv::Mat colwiseCum = cv::Mat::zeros(1, centroid.cols, CV_32F);

// Compute column wise cumulative sum
	cv::reduce(descriptors, colwiseCum, 0, CV_REDUCE_SUM, CV_32F);

// Threshold the resulting cumulative sum vector
	cv::threshold(colwiseCum, centroid, centroid.rows / 2, 1, CV_THRESH_BINARY);
}
