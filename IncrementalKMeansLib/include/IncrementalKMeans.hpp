/*
 * IncrementalKMeans.h
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#ifndef INCREMENTALKMEANS_H_
#define INCREMENTALKMEANS_H_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/flann/flann.hpp>

#include <DynamicMat.hpp>
#include <VocabBase.hpp>

namespace vlr {

struct IncrementalKMeansParams: public cvflann::IndexParams {
	IncrementalKMeansParams(int numClusters = 1000000) {
		(*this)["num.clusters"] = numClusters;
	}
};

class IncrementalKMeans: public VocabBase {

protected:

	// Dimensionality of the data under clustering (in Bytes) (d)
	int m_dim;
	// Number of data instances (n)
	int m_numDatapoints;
	// Number of clusters (k)
	int m_numClusters;

	// Reference to the matrix with data to cluster (D)
	vlr::Mat m_dataset;
	// Matrix of clusters centers (C)
	cv::Mat m_centroids;
	// Clusters variance matrices (R)
	cv::Mat m_clustersVariances;
	// Weights for each cluster (W)
	cv::Mat m_clustersWeights;
	// Sum of points per cluster (M)
	cv::Mat m_clustersSums;
	// Number of data points assigned to each cluster (N)
	cv::Mat m_clustersCounts;
	//
	cv::Mat m_clusterDistances;
	// Distance from each cluster center to the null transaction (Delta)
	cv::Mat m_clusterDistancesToNullTransaction;

	cv::Mat m_miu;
	cv::Mat m_sigma;

	// List of outliers
	std::vector<std::pair<int, double>> m_outliers;

public:

	/**
	 * Class constructor.
	 */
	IncrementalKMeans(vlr::Mat data, const cvflann::IndexParams& params = IncrementalKMeansParams());

	/**
	 * Class destroyer.
	 */
	virtual ~IncrementalKMeans();

	void build();

	void save(const std::string& filename) const;

	void load(const std::string& filename);

	size_t size() const {
		return m_centroids.rows;
	}

	/**
	 * Getters.
	 */

	const cv::Mat& getCentroids() const {
		return m_centroids;
	}

	const cv::Mat& getClustersCounts() const {
		return m_clustersCounts;
	}

	const cv::Mat& getClustersSums() const {
		return m_clustersSums;
	}

	const cv::Mat& getClustersVariances() const {
		return m_clustersVariances;
	}

	const cv::Mat& getClustersWeights() const {
		return m_clustersWeights;
	}

	const vlr::Mat& getDataset() const {
		return m_dataset;
	}

	int getDim() const {
		return m_dim;
	}

	const cv::Mat& getMiu() const {
		return m_miu;
	}

	int getNumClusters() const {
		return m_numClusters;
	}

	int getNumDatapoints() const {
		return m_numDatapoints;
	}

	const std::vector<std::pair<int, double> >& getOutliers() const {
		return m_outliers;
	}

	const cv::Mat& getSigma() const {
		return m_sigma;
	}

protected:

	/**
	 * Initializes cluster centers using a mean-based strategy.
	 */
	void initCentroids();

	/**
	 * Determine cluster membership for the given transaction by means of
	 * sparse distance computation.
	 *
	 * @param transaction - A transaction
	 * @param clusterIndex - Index to the closest cluster center
	 * @param distanceToCluster - Distance to the closest cluster center
	 */
	void findNearestNeighbor(cv::Mat transaction, int& clusterIndex, double& distanceToCluster);

	/**
	 * Compute clusters centers using sufficient statistics as proposed by Ordonez2003.
	 *
	 * @param i - Number of processed transactions
	 */
	void computeCentroids(const int& i);

	/**
	 * Pre-compute distances between the null transaction and all the centers.
	 */
	void preComputeDistances();

	/**
	 * Insert outliers in an ascending manner.
	 *
	 * @param transactionIndex
	 * @param distanceToCluster
	 */
	void insertOutlier(const int& transactionIndex, const double& distanceToCluster);

	/**
	 * Fills empty clusters by re-assigning in descending order the outlier transactions.
	 */
	void handleEmptyClusters();

	/**
	 * Sum the given transaction to the given row in the cluster sum matrix.
	 *
	 * @param transaction
	 * @param rowIndex
	 */
	void sparseSum(cv::Mat transaction, const int& rowIndex);

	/**
	 * Subtract the given transaction to the given row in the cluster sum matrix.
	 *
	 * @param transaction
	 * @param rowIndex
	 */
	void sparseSubtraction(cv::Mat transaction, const int& rowIndex);

};

} /* namespace vlr */

#endif /* INCREMENTALKMEANS_H_ */
