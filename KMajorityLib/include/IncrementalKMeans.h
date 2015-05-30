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
//		int maxIterations = 10, vlr::indexType nnType = vlr::HIERARCHICAL, cvflann::flann_centers_init_t centersInitMethod = cvflann::FLANN_CENTERS_RANDOM
		(*this)["num.clusters"] = numClusters;
//		(*this)["max.iterations"] = maxIterations;
//		(*this)["centers.init.method"] = centersInitMethod;
//		(*this)["nn.type"] = nnType;
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
	cv::Mat m_dataset;
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

	cv::Mat m_miu;
	cv::Mat m_sigma;

	std::vector<std::pair<int, double>> m_outliers;

public:

	/**
	 * Class constructor.
	 */
	IncrementalKMeans(cv::Mat data, const cvflann::IndexParams& params = IncrementalKMeansParams());

	/**
	 * Class destroyer.
	 */
	virtual ~IncrementalKMeans();

	void build();

	void save(const std::string& filename) const;

	void load(const std::string& filename);

	size_t size() const {
		return 0;
	}

private:

	/**
	 * Finds the closest cluster center to the given transaction.
	 *
	 * @param transaction
	 * @return
	 */
	int findNearestNeighbor(cv::Mat transaction);

};

} /* namespace vlr */

#endif /* INCREMENTALKMEANS_H_ */
