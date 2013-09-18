/*
 * kmajority_index.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <kmajority_index.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

#include <iostream>
#include <functional>
#include <bitset>

typedef cvflann::Hamming<uchar> Distance;
typedef typename Distance::ResultType DistanceType;

void KMajorityIndex::cluster(const cv::Mat& descriptors) {

	if (descriptors.type() != CV_8U) {
		fprintf(stderr,
				"KMajorityIndex::cluster: error, descriptors matrix is not binary");
		return;
	}

	if (descriptors.empty()) {
		fprintf(stderr, "KMajorityIndex::cluster: error, descriptors is empty");
		return;
	}

	this->n = descriptors.rows;
	this->dim = descriptors.cols;

	this->belongs_to = new unsigned int[this->n];
	// Initially all transactions belong to any cluster
	std::fill_n(this->belongs_to, this->n, this->k);

	this->distance_to = new unsigned int[this->n];
	// Initially all transactions are at the farthest possible distance
	// i.e. dim*8 the max hamming distance
	std::fill_n(this->distance_to, this->n, this->dim * 8);

	this->cluster_counts = new unsigned int[this->k];
	// Initially no transaction is assigned to any cluster
	std::fill_n(this->cluster_counts, this->k, 0);

	// Trivial case: less data than clusters, assign one to each cluster
	if (this->n <= this->k) {
		centroids.create(this->k, dim, descriptors.type());
		for (unsigned int i = 0; i < this->n; ++i) {
			descriptors.row(i).copyTo(
					centroids(cv::Range(i, i + 1), cv::Range(0, this->dim)));
		}
		return;
	}

	// Randomly generate clusters
	this->initCentroids(descriptors);

	// Assign data to clusters
	this->quantize(descriptors);

	bool converged = false;

	unsigned int iteration = 0;
	while (converged == false && iteration < max_iterations) {
		iteration++;
		// Compute the new cluster centers
		this->computeCentroids(descriptors);

		// Reassign data to clusters
		converged = this->quantize(descriptors);

		// TODO handle empty clusters case
		// Find empty clusters
//		for (unsigned int j = 0; j < k; j++) {
//			if (cluster_counts[j] == 0) {
//				printf("Cluster %u is empty\n", j);
//				// Find farthest element to its assigned cluster
//				unsigned int farthest_element_idx = 0;
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

void KMajorityIndex::initCentroids(const cv::Mat& descriptors) {

	// Initializing variables useful for obtaining indexes of random chosen center
	cv::Ptr<int> centers_idx = new int[k];
	int centers_length;
	cv::Ptr<int> indices = new int[n];
	for (unsigned int i = 0; i < this->n; i++) {
		indices[i] = i;
	}

	// Randomly chose centers
	chooseCentersRandom(k, indices, n, centers_idx, centers_length);

	// Assign centers based on the chosen indexes
	centroids.create(centers_length, dim, descriptors.type());
	for (int i = 0; i < centers_length; i++) {
		descriptors.row(centers_idx[i]).copyTo(
				centroids(cv::Range(i, i + 1), cv::Range(0, this->dim)));
	}

}

bool KMajorityIndex::quantize(const cv::Mat& descriptors) {

	bool converged = true;

	// Comparison of all descriptors vs. all centroids
	for (unsigned int i = 0; i < this->n; i++) {
		// Set minimum distance as the distance to its assigned cluster
		int min_hd = distance_to[i];

		for (unsigned int j = 0; j < this->k; j++) {
			// Compute hamming distance between ith descriptor and jth cluster
			cv::Hamming distance;
			int hd = distance(descriptors.row(i).data, centroids.row(j).data,
					descriptors.cols);

			// Update cluster assignment to the nearest cluster
			if (hd < min_hd) {
				min_hd = hd;
				// If cluster assignment changed that means the algorithm hasn't converged yet
				if (belongs_to[i] != j) {
					converged = false;
				}
				// Decrease cluster count in case it was assigned to some valid cluster before.
				// Recall that initially all transaction are assigned to kth cluster which
				// is not valid since valid clusters run from 0 to k-1 both inclusive.
				if (belongs_to[i] != k) {
					cluster_counts[belongs_to[i]]--;
				}
				belongs_to[i] = j;
				cluster_counts[j]++;
				distance_to[i] = hd;
			}
		}
	}

	return converged;
}

void KMajorityIndex::computeCentroids(const cv::Mat& descriptors) {

	// Warning: using matrix of integers, there might be an overflow when summing too much descriptors
	cv::Mat bitwiseCount(1, this->dim * 8, cv::DataType<int>::type);
	// Loop over all clusters
	for (unsigned int j = 0; j < k; j++) {
		// Zeroing all cumulative variable dimension
		bitwiseCount(cv::Range(0, 1), cv::Range(0, bitwiseCount.cols)) =
				cv::Scalar::all(0);
		// Zeroing all the centroid dimensions
		centroids(cv::Range(j, j + 1), cv::Range(0, centroids.cols)) =
				cv::Scalar::all(0);
		// Loop over all data
		for (unsigned int i = 0; i < this->n; i++) {
			// Finding all data assigned to jth clusther
			if (belongs_to[i] == j) {
				uchar byte;
				for (int l = 0; l < bitwiseCount.cols; l++) {
					// bit: 7-(l%8) col: (int)l/8 descriptor: i
					// Load byte every 8 bits
					if ((l % 8) == 0) {
						byte = *(descriptors.row(i).col((int) l / 8).data);
					}
					// Warning: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
					// bit at ith position is mod(bitleftshift(byte,i),2) where ith position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
					bitwiseCount.at<int>(0, l) +=
							((int) ((byte >> (7 - (l % 8))) % 2));
				}
			}
		}

		// In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
		for (int l = 0; l < bitwiseCount.cols; l++) {
			// If the bitcount for jth cluster at dimension l is greater than half of the data assigned to it
			// then set lth centroid bit to 1 otherwise set it to 0 (break ties randomly)
			bool bit;
			// There is a tie if the number of data assigned to jth cluster is even
			// AND the number of bits set to 1 in lth dimension is the half of the data assigned to jth cluster
			if (this->cluster_counts[j] % 2 == 1
					&& 2 * bitwiseCount.at<int>(0, l)
							== (int) this->cluster_counts[j]) {
				bit = rand() % 2;
			} else {
				bit = 2 * bitwiseCount.at<int>(0, l)
						> (int) (this->cluster_counts[j]);
			}
			centroids.at<unsigned char>(j,
					(int) (bitwiseCount.cols - 1 - l) / 8) += (bit)
					<< ((bitwiseCount.cols - 1 - l) % 8);
		}
	}
}
