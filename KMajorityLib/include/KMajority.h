/*
 * KMajority.h
 *
 *  Created on: Aug 28, 2013
 *      Author: andresf
 */

#ifndef KMAJORITY_INDEX_H_
#define KMAJORITY_INDEX_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>

#include <DynamicMat.hpp>
#include <VocabBase.hpp>

typedef cvflann::Hamming<uchar> Distance;
typedef typename Distance::ResultType DistanceType;

namespace vlr {

// Allowed nearest neighbor index algorithms
enum indexType {
	LINEAR = 0, HIERARCHICAL = 1
};

cvflann::NNIndex<Distance>* createIndexByType(
		const cvflann::Matrix<typename Distance::ElementType>& dataset,
		vlr::indexType type, const cvflann::IndexParams& params);

struct KMajorityParams: public cvflann::IndexParams {
	KMajorityParams(int numClusters = 1000000, int maxIterations = 10,
			vlr::indexType nnType = vlr::HIERARCHICAL,
			cvflann::flann_centers_init_t centersInitMethod =
					cvflann::FLANN_CENTERS_RANDOM) {
		(*this)["num.clusters"] = numClusters;
		(*this)["max.iterations"] = maxIterations;
		(*this)["centers.init.method"] = centersInitMethod;
		(*this)["nn.type"] = nnType;
	}
};

class KMajority: public VocabBase {

protected:

	// Number of clusters
	int m_numClusters;
	// Maximum number of iterations
	int m_maxIterations;
	// Method for initializing centers
	cvflann::flann_centers_init_t m_centersInitMethod;
	// Reference to the matrix with data to cluster
	vlr::Mat& m_dataset;
	// Dimensionality of the data under clustering (in Bytes)
	int m_dim;
	// Number of data instances
	int m_numDatapoints;
	// List of the cluster each data point belongs to
	std::vector<int> m_belongsTo;
	// List of distance from each data point to the cluster it belongs to
	std::vector<DistanceType> m_distanceTo;
	// Number of data points assigned to each cluster
	std::vector<int> m_clusterCounts;
	// Matrix of clusters centers
	cv::Mat m_centroids;
	// Nearest neighbor index type
	vlr::indexType m_nnType;
	// Index for addressing nearest neighbors search
	cvflann::NNIndex<Distance>* m_nnIndex = NULL;
	// Nearest neighbors index parameters
	cvflann::IndexParams m_nnIndexParams;

public:

	/**
	 * Class constructor.
	 *
	 * @param inputData - Reference to the matrix with the data to be clustered
	 * @param params - Parameters to the k-majority algorithm
	 * @param nnIndexParams - Parameters to the nearest neighbors index
	 */
	KMajority(vlr::Mat& inputData = vlr::DEFAULT_INPUTDATA,
			const cvflann::IndexParams& params = KMajorityParams(),
			const cvflann::IndexParams& nnIndexParams =
					cvflann::HierarchicalClusteringIndexParams());

	/**
	 * Class destroyer.
	 */
	~KMajority();

	/**
	 * Implements k-means clustering loop.
	 */
	void build();

	/**
	 * Saves the vocabulary to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the vocabulary
	 */
	void save(const std::string& filename) const;

	/**
	 * Loads the vocabulary to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the vocabulary
	 */
	void load(const std::string& filename);

	size_t size() const {
		return m_centroids.rows;
	}

	/**
	 * Decomposes data into bits and accumulates them into cumResult.
	 *
	 * @param data - Row vector containing the data to accumulate
	 * @param accVector - Row oriented accumulator vector
	 */
	static void cumBitSum(const cv::Mat& data, cv::Mat& accVector);

	/**
	 * Component wise thresholding of accumulator vector.
	 *
	 * @param accVector - Row oriented accumulator vector
	 * @param result - Row vector containing the thresholding result
	 * @param threshold - Threshold value, typically the number of data points used to compute the accumulator vector
	 */
	static void majorityVoting(const cv::Mat& accVector, cv::Mat& result,
			const int& threshold);

	/**** Getters ****/

	const cv::Mat& getCentroids() const;

	const std::vector<int>& getClusterCounts() const;

	const std::vector<int>& getClusterAssignments() const;

private:

	/**
	 * Initializes cluster centers choosing among the data points indicated by indices.
	 */
	void initCentroids();

	/**
	 * Implements majority voting scheme for cluster centers computation
	 * based on component wise majority of bits from data matrix
	 * as proposed by Grana2013.
	 */
	void computeCentroids();

	/**
	 * Assigns data to clusters by means of Hamming distance.
	 *
	 * @return true if convergence was achieved (cluster assignment didn't changed), false otherwise
	 */
	bool quantize();

	/**
	 * Fills empty clusters using data assigned to the most populated ones.
	 */
	void handleEmptyClusters();

	/**
	 * Build index for addressing nearest neighbors descriptors search.
	 */
	void updateIndex();

};

} /* namespace vlr */

#endif /* KMAJORITY_H_ */
