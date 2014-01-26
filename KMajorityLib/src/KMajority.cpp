/*
 * KMajority.cpp
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#include <KMajority.h>
#include <CentersChooser.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/flann/random.h>
#include <opencv2/flann/dist.h>

#include <iostream>
#include <functional>
#include <bitset>

KMajority::KMajority(int numClusters, int maxIterations, const cv::Mat& data,
		vlr::indexType nnMethod,
		cvflann::flann_centers_init_t centersInitMethod) :
		m_numClusters(numClusters), m_maxIterations(maxIterations), m_nnMethod(
				nnMethod), m_centersInitMethod(centersInitMethod), m_dataset(
				data), m_dim(data.cols), m_nnIndex(NULL) {

	m_numDatapoints = m_dataset.rows;

	// Initially all transactions belong to any cluster
	m_belongsTo.clear();
	m_belongsTo.resize(m_dataset.rows, numClusters);

	// Initially all transactions are at the farthest possible distance
	// i.e. m_dim*8 the max Hamming distance
	m_distanceTo.clear();
	m_distanceTo.resize(m_dataset.rows, data.cols * 8);

	// Initially no transaction is assigned to any cluster
	m_clusterCounts.clear();
	m_clusterCounts.resize(numClusters, 0);

}

KMajority::~KMajority() {
	delete m_nnIndex;
}

void KMajority::cluster() {

	if (m_dataset.type() != CV_8U) {
		throw std::runtime_error(
				"[KMajority::cluster] Descriptors matrix is not binary");
	}

	if (m_dataset.empty()) {
		throw std::runtime_error("[KMajority::cluster] Descriptors is empty");
	}

	// Trivial case: less data than clusters, assign one data point per cluster
	if (m_numDatapoints <= m_numClusters) {
		m_centroids.create(m_numClusters, m_dim, m_dataset.type());
		for (int i = 0; i < m_numDatapoints; ++i) {
			m_dataset.row(i).copyTo(
					m_centroids(cv::Range(i, i + 1), cv::Range(0, m_dim)));
			m_belongsTo[i] = i;
		}
		return;
	}

	// Randomly generate clusters
	initCentroids();

	// Assign data to clusters
	quantize();

	bool converged = false;
	int iteration = 0;

	while (converged == false && iteration < m_maxIterations) {

		++iteration;

		// Compute the new cluster centers
		computeCentroids();

		// Reassign data to clusters
		converged = quantize();

		// Handle empty clusters case
		handleEmptyClusters();
	}

}

void KMajority::initCentroids() {

	// Initializing variables useful for obtaining indexes of random chosen center
	std::vector<int> centers_idx(m_numClusters);
	int centers_length;

	// Array of indices indicating data points involved in the clustering process
	int* indices = new int[m_dataset.rows];
	for (int i = 0; i < m_dataset.rows; ++i) {
		indices[i] = i;
	}

	// Randomly chose centers
	CentersChooser<Distance::ElementType, cv::Hamming>::create(
			m_centersInitMethod)->chooseCenters(m_numClusters, indices,
			m_numDatapoints, centers_idx, centers_length, m_dataset);

	delete[] indices;

	// Assign centers based on the chosen indexes
	m_centroids.create(centers_length, m_dim, m_dataset.type());
	for (int i = 0; i < centers_length; ++i) {
		m_dataset.row(centers_idx[i]).copyTo(
				m_centroids(cv::Range(i, i + 1), cv::Range(0, m_dim)));
	}

	cvflann::IndexParams m_indexParams =
			cvflann::HierarchicalClusteringIndexParams();

	// Build index for addressing nearest neighbors descriptors search

	m_nnIndex = vlr::createIndexByType(
			cvflann::Matrix<Distance::ElementType>(m_centroids.data,
					m_centroids.rows, m_centroids.cols), Distance(),
			m_nnMethod);

	m_nnIndex->buildIndex();

}

bool KMajority::quantize() {

	bool converged = true;

	// Number of nearest neighbors
	int knn = 1;

	// The indices of the nearest neighbors found (numQueries X numNeighbors)
	cvflann::Matrix<int> indices(new int[1 * knn], 1, knn);

	// Distances to the nearest neighbors found (numQueries X numNeighbors)
	cvflann::Matrix<DistanceType> distances(new int[1 * knn], 1, knn);

	for (int i = 0; i < m_numDatapoints; ++i) {
		std::fill(indices.data, indices.data + indices.rows * indices.cols, 0);
		std::fill(distances.data,
				distances.data + distances.rows * distances.cols, 0.0f);

		/* Get new cluster it belongs to */
		m_nnIndex->knnSearch(
				cvflann::Matrix<Distance::ElementType>(m_dataset.row(i).data, 1,
						m_dataset.cols), indices, distances, knn,
				cvflann::SearchParams());

		/* Check if it changed */
		// If cluster assignment changed that means the algorithm hasn't converged yet
		if (m_belongsTo[i] != indices[0][0]) {
			converged = false;
		}

		/* Update cluster assignment and cluster counts */
		// Decrease cluster count in case it was assigned to some valid cluster before.
		// Recall that initially all transaction are assigned to kth cluster which
		// is not valid since valid clusters run from 0 to k-1 both inclusive.
		if (m_belongsTo[i] != m_numClusters) {
			--m_clusterCounts[m_belongsTo[i]];
		}
		m_belongsTo[i] = indices[0][0];
		++m_clusterCounts[indices[0][0]];
		m_distanceTo[i] = distances[0][0];
	}

	return converged;
}

void KMajority::computeCentroids() {

	// Warning: using matrix of integers, there might be an overflow when summing too much descriptors
	cv::Mat bitwiseCount(m_numClusters, m_dim * 8, cv::DataType<int>::type);
	// Zeroing matrix of cumulative bits
	bitwiseCount = cv::Scalar::all(0);
	// Zeroing all cluster centers dimensions
	m_centroids = cv::Scalar::all(0);

	// Bitwise summing the data into each center
	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat b = bitwiseCount.row(m_belongsTo[i]);
		KMajority::cumBitSum(m_dataset.row(i), b);
	}

	// Bitwise majority voting
	for (int j = 0; j < m_numClusters; j++) {
		cv::Mat centroid = m_centroids.row(j);
		KMajority::majorityVoting(bitwiseCount.row(j), centroid,
				m_clusterCounts[j]);
	}
}

void KMajority::cumBitSum(const cv::Mat& data, cv::Mat& accVector) {

	// cumResult and data must be row vectors
	if (data.rows != 1 || accVector.rows != 1) {
		throw std::runtime_error(
				"[KMajority::cumBitSum] data and cumResult parameters must be row vectors\n");
	}
	// cumResult and data must be same length
	if (data.cols * 8 != accVector.cols) {
		throw std::runtime_error(
				"[KMajority::cumBitSum] number of columns in cumResult must be that of data times 8\n");
	}

	uchar byte = 0;
	for (int l = 0; l < accVector.cols; l++) {
		// bit: 7-(l%8) col: (int)l/8 descriptor: i
		// Load byte every 8 bits
		if ((l % 8) == 0) {
			byte = *(data.col((int) l / 8).data);
		}
		// Note: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
		// bit at ith position is mod(bitleftshift(byte,i),2) where ith position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
		accVector.at<int>(0, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
	}

}

void KMajority::majorityVoting(const cv::Mat& accVector, cv::Mat& result,
		const int& threshold) {

	// cumResult and data must be a row vectors
	if (accVector.rows != 1 || result.rows != 1) {
		throw std::runtime_error(
				"[KMajority::majorityVoting] `bitwiseCount` and `centroid` parameters must be row vectors\n");
	}

	// cumResult and data must be same length
	if (result.cols * 8 != accVector.cols) {
		throw std::runtime_error(
				"[KMajority::majorityVoting] number of columns in `bitwiseCount` must be that of `data` times 8\n");
	}

	// In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
	for (int l = 0; l < accVector.cols; ++l) {
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

void KMajority::handleEmptyClusters() {

	// If some cluster appeared to be empty then:
	// 1. Find the biggest cluster.
	// 2. Find farthest point in the biggest cluster
	// 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster.

	for (int k = 0; k < m_numClusters; ++k) {
		if (m_clusterCounts[k] != 0) {
			continue;
		}

		// 1. Find the biggest cluster
		int max_k = 0;
		for (int k1 = 1; k1 < m_numClusters; ++k1) {
			if (m_clusterCounts[max_k] < m_clusterCounts[k1])
				max_k = k1;
		}

		// 2. Find farthest point in the biggest cluster
		DistanceType maxDist(-1);
		int idxFarthestPt = -1;
		for (int i = 0; i < m_numDatapoints; ++i) {
			if (m_belongsTo[i] == max_k) {
				if (maxDist < m_distanceTo[i]) {
					maxDist = m_distanceTo[i];
					idxFarthestPt = i;
				}
			}
		}

		// 3. Exclude the farthest point from the biggest cluster and form a new 1-point cluster
		--m_clusterCounts[max_k];
		++m_clusterCounts[k];
		m_belongsTo[idxFarthestPt] = k;
	}
}

const cv::Mat& KMajority::getCentroids() const {
	return m_centroids;
}

const std::vector<int>& KMajority::getClusterCounts() const {
	return m_clusterCounts;
}

const std::vector<int>& KMajority::getClusterAssignments() const {
	return m_belongsTo;
}

cvflann::NNIndex<Distance>* vlr::createIndexByType(
		const cvflann::Matrix<typename Distance::ElementType>& dataset,
		const Distance& distance, vlr::indexType type) {

	cvflann::IndexParams params;
	cvflann::NNIndex<Distance>* nnIndex;

	switch (type) {
	case vlr::indexType::LINEAR:
		printf("Creating [Linear] index\n");
		params = cvflann::LinearIndexParams();
		nnIndex = new cvflann::LinearIndex<Distance>(dataset, params, distance);
		break;
	case vlr::indexType::HIERARCHICAL:
		printf("Creating [HierarchicalClustering] index\n");
		params = cvflann::HierarchicalClusteringIndexParams();
		nnIndex = new cvflann::HierarchicalClusteringIndex<Distance>(dataset,
				params, distance);
		break;
	default:
		throw std::runtime_error("Unknown index type");
	}

	return nnIndex;
}
