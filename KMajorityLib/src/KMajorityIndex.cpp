/*
 * kmajority_index.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <KMajorityIndex.h>
#include <CentersChooser.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

#include <iostream>
#include <functional>
#include <bitset>

typedef cvflann::Hamming<uchar> Distance;
typedef typename Distance::ResultType DistanceType;

KMajorityIndex::KMajorityIndex(uint _k, uint _max_iterations,
		const cv::Mat& _data, cv::Ptr<int>& _indices,
		const int& _indices_length, uint* _belongs_to,
		cvflann::flann_centers_init_t centers_init) :
		k(_k), max_iterations(_max_iterations), m_centers_init(centers_init), data(
				_data), indices(_indices), indices_length(_indices_length), dim(
				_data.cols), n(_indices_length), distance_to(
				new uint[_indices_length]), cluster_counts(new uint[_k]) {

	belongs_to = _belongs_to != NULL ? _belongs_to : new uint[_indices_length];
	// Set the pointer belongs_to to be deleted
	delete_belongs_to = _belongs_to == NULL;

	// Initially all transactions belong to any cluster
	std::fill_n(this->belongs_to, _indices_length, _k);

	// Initially all transactions are at the farthest possible distance
	// i.e. dim*8 the max hamming distance
	std::fill_n(this->distance_to, _indices_length, _data.cols * 8);

	// Initially no transaction is assigned to any cluster
	std::fill_n(this->cluster_counts, this->k, 0);
}

KMajorityIndex::~KMajorityIndex() {
	if (delete_belongs_to == true) {
		delete[] belongs_to;
	}
	delete[] distance_to;
	delete[] cluster_counts;
}

void KMajorityIndex::cluster() {
	if (data.type() != CV_8U) {
		throw std::runtime_error(
				"KMajorityIndex::cluster: error, descriptors matrix is not binary");
	}

	if (data.empty()) {
		throw std::runtime_error(
				"KMajorityIndex::cluster: error, descriptors is empty");
	}

	// Trivial case: less data than clusters, assign one data point per cluster
	if (this->n <= this->k) {
		centroids.create(this->k, dim, data.type());
		for (uint i = 0; i < this->n; ++i) {
			data.row(i).copyTo(
					centroids(cv::Range(i, i + 1), cv::Range(0, this->dim)));
			belongs_to[i] = i;
		}
		return;
	}

	// Randomly generate clusters
	this->initCentroids();

	// Assign data to clusters
	this->quantize();

	bool converged = false;
	uint iteration = 0;

	while (converged == false && iteration < max_iterations) {

		iteration++;

		// Compute the new cluster centers
		this->computeCentroids();

		// Reassign data to clusters
		converged = this->quantize();

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

void KMajorityIndex::initCentroids() {

	// Initializing variables useful for obtaining indexes of random chosen center
	cv::Ptr<int> centers_idx = new int[k];
	int centers_length;

	// Randomly chose centers
	CentersChooser<cvflann::Hamming<uchar> >::create(m_centers_init)->chooseCenters(
			k, indices, n, centers_idx, centers_length, this->data);

	// Assign centers based on the chosen indexes
	centroids.create(centers_length, dim, data.type());
	for (int i = 0; i < centers_length; i++) {
		data.row(centers_idx[i]).copyTo(
				centroids(cv::Range(i, i + 1), cv::Range(0, this->dim)));
	}

}

bool KMajorityIndex::quantize() {

	bool converged = true;

	// Comparison of all descriptors vs. all centroids
	for (size_t i = 0; i < this->n; i++) {
		// Set minimum distance as the distance to its assigned cluster
		int min_hd = distance_to[i];

		for (size_t j = 0; j < this->k; j++) {
			// Compute hamming distance between ith descriptor and jth cluster
			cv::Hamming distance;
			int hd = distance(data.row(indices[i]).data, centroids.row(j).data,
					data.cols);

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

void KMajorityIndex::computeCentroids() {

	// Warning: using matrix of integers, there might be an overflow when summing too much descriptors
	cv::Mat bitwiseCount(this->k, this->dim * 8, cv::DataType<int>::type);
	// Zeroing matrix of cumulative bits
	bitwiseCount = cv::Scalar::all(0);
	// Zeroing all the centroids dimensions
	centroids = cv::Scalar::all(0);

	// Bitwise summing the data into each centroid
	for (size_t i = 0; i < this->n; i++) {
		uint j = belongs_to[i];
		cv::Mat b = bitwiseCount.row(j);
		KMajorityIndex::cumBitSum(data.row(i), b);
	}

	// Bitwise majority voting
	for (size_t j = 0; j < k; j++) {
		cv::Mat centroid = centroids.row(j);
		KMajorityIndex::majorityVoting(bitwiseCount.row(j), centroid,
				cluster_counts[j]);
	}
}

void KMajorityIndex::cumBitSum(const cv::Mat& data, cv::Mat& accVector) {

	// cumResult and data must be a row vectors
	if (data.rows != 1 || accVector.rows != 1) {
		throw std::runtime_error(
				"KMajorityIndex::cumBitSum: data and cumResult parameters must be row vectors\n");
	}
	// cumResult and data must be same length
	if (data.cols * 8 != accVector.cols) {
		throw std::runtime_error(
				"KMajorityIndex::cumBitSum: number of columns in cumResult must be that of data times 8\n");
	}

	uchar byte = 0;
	for (int l = 0; l < accVector.cols; l++) {
		// bit: 7-(l%8) col: (int)l/8 descriptor: i
		// Load byte every 8 bits
		if ((l % 8) == 0) {
			byte = *(data.col((int) l / 8).data);
		}
		// Warning: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
		// bit at ith position is mod(bitleftshift(byte,i),2) where ith position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
		accVector.at<int>(0, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
	}

}

void KMajorityIndex::majorityVoting(const cv::Mat& accVector, cv::Mat& result,
		const uint& threshold) {

	// cumResult and data must be a row vectors
	if (accVector.rows != 1 || result.rows != 1) {
		throw std::runtime_error(
				"KMajorityIndex::majorityVoting: `bitwiseCount` and `centroid` parameters must be row vectors\n");
	}

	// cumResult and data must be same length
	if (result.cols * 8 != accVector.cols) {
		throw std::runtime_error(
				"KMajorityIndex::majorityVoting: number of columns in `bitwiseCount` must be that of `data` times 8\n");
	}

	// In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
	for (size_t l = 0; (int) l < accVector.cols; l++) {
		// If the bitcount for jth cluster at dimension l is greater than half of the data assigned to it
		// then set lth centroid bit to 1 otherwise set it to 0 (break ties randomly)
		bool bit;
		// There is a tie if the number of data assigned to jth cluster is even
		// AND the number of bits set to 1 in lth dimension is the half of the data assigned to jth cluster
		if (threshold % 2 == 1
				&& 2 * accVector.at<int>(0, l) == (int) threshold) {
			bit = rand() % 2;
		} else {
			bit = 2 * accVector.at<int>(0, l) > (int) (threshold);
		}
		// Stores the majority voting result from the LSB to the MSB
		result.at<unsigned char>(0, (int) (accVector.cols - 1 - l) / 8) += (bit)
				<< ((accVector.cols - 1 - l) % 8);
	}
}

const cv::Mat& KMajorityIndex::getCentroids() const {
	return centroids;
}

uint* KMajorityIndex::getClusterCounts() const {
	return cluster_counts;
}

uint* KMajorityIndex::getClusterAssignments() const {
	return belongs_to;
}

int KMajorityIndex::getNumberOfClusters() const {
	return centroids.rows;
}
