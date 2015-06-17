/*
 * IncrementalKMeans.cpp
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#include <IncrementalKMeans.hpp>
#include <opencv2/core/core.hpp>

namespace vlr {

IncrementalKMeans::IncrementalKMeans(vlr::Mat data, const cvflann::IndexParams& params) :
		m_dataset(data), m_dim(data.cols), m_numDatapoints(data.rows) {

	// Attributes initialization
	m_numClusters = cvflann::get_param<int>(params, "num.clusters");

	// Compute the global data set mean
	m_miu = cv::Mat::zeros(1, m_dim * 8, cv::DataType<double>::type);
	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat row = m_dataset.row(i);
		uchar byte = 0;
		for (int l = 0; l < m_miu.cols; l++) {
			if ((l % 8) == 0) {
				byte = *(row.col((int) l / 8).data);
			}
			m_miu.at<double>(0, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
		}
	}
	m_miu /= m_numDatapoints;

	// Compute the global data set standard deviations
	m_sigma = cv::Mat::zeros(1, m_dim * 8, cv::DataType<double>::type);
	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat row = m_dataset.row(i);
		uchar byte = 0;
		for (int l = 0; l < m_sigma.cols; l++) {
			if ((l % 8) == 0) {
				byte = *(row.col((int) l / 8).data);
			}
			m_sigma.at<double>(0, l) += pow(((int) ((byte >> (7 - (l % 8))) % 2)) - m_miu.at<double>(0, l), 2);
		}
	}
	m_sigma /= m_numDatapoints;
	cv::sqrt(m_sigma, m_sigma);

	m_centroids.create(m_numClusters, m_dim * 8, cv::DataType<double>::type);
	m_clustersVariances.create(m_numClusters, m_dim * 8, cv::DataType<double>::type);
	m_clustersWeights.create(1, m_numClusters, cv::DataType<double>::type);
	m_clustersSums.create(m_numClusters, m_dim * 8, cv::DataType<int>::type);
	m_clustersCounts.create(1, m_numClusters, cv::DataType<int>::type);
	m_clusterDistances.create(1, m_numClusters, cv::DataType<double>::type);
	m_clusterDistancesToNullTransaction.create(1, m_numClusters, cv::DataType<double>::type);

}

// --------------------------------------------------------------------------

IncrementalKMeans::~IncrementalKMeans() {
}

// --------------------------------------------------------------------------

void IncrementalKMeans::build() {

	// Mean-based initialization
	srand(time(NULL));
	for (int j = 0; j < m_numClusters; ++j) {
		// Cj <- miu +/-sigma*r/d
		double r = (double) rand() / RAND_MAX;
		if (rand() % 2 == 0) {
			m_centroids.row(j) = (m_miu + m_sigma * r / (m_dim * 8));
		} else {
			m_centroids.row(j) = (m_miu - m_sigma * r / (m_dim * 8));
		}
	}
	preComputeDistances();
	// Nj <- 0
	m_clustersCounts = cv::Mat::zeros(1, m_numClusters, cv::DataType<int>::type);
	// Mj <- 0
	m_clustersSums = cv::Mat::zeros(m_numClusters, m_dim * 8, cv::DataType<int>::type);
	// Wj <- 1/k
	m_clustersWeights = cv::Mat::ones(1, m_numClusters, cv::DataType<double>::type) / m_numClusters;

	double L = sqrt(m_numDatapoints);
	int clusterIndex;
	double distanceToCluster;

	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat transaction = m_dataset.row(i);
		// Cluster assignment
		// j = NN(ti)
		findNearestNeighbor(transaction, clusterIndex, distanceToCluster);
		// Mj <- Mj + ti
		uchar byte = 0;
		for (int l = 0; l < m_clustersSums.cols; l++) {
			if ((l % 8) == 0) {
				byte = *(transaction.col((int) l / 8).data);
			}
			m_clustersSums.at<int>(clusterIndex, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
		}
		// Nj <- Nj + 1
		m_clustersCounts.col(clusterIndex) += 1;
		// Insert outliers in an ordered manner
		std::pair<int, double> item(i, distanceToCluster);
		m_outliers.insert(std::upper_bound(m_outliers.begin(), m_outliers.end(), item, [](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) {return lhs.second < rhs.second;}), item);

		// Update clusters centers every L times
		if (i % ((int) (m_numDatapoints / L)) == 0) {
			// Re-compute clusters centers
			for (int j = 0; j < m_numClusters; ++j) {
				// Cj <- Mj/Nj
				m_centroids.row(clusterIndex) = m_clustersSums.row(clusterIndex) / m_clustersCounts.col(clusterIndex);
				// Rj <- Cj - diag(Cj*Cj')
				cv::Mat clusterVariance(m_dim * 8, m_dim * 8, CV_32F);
				cv::mulTransposed(m_centroids.row(clusterIndex), clusterVariance, true);
				m_clustersVariances.row(clusterIndex) = m_centroids.row(clusterIndex) - clusterVariance.diag(0);
				// Wj <- Nj/i
				m_clustersWeights.col(clusterIndex) = m_clustersCounts.col(clusterIndex) / i;
			}
			preComputeDistances();
			// Re-seeding
			for (int j = 0; j < m_numClusters; ++j) {
				if (m_outliers.empty()) {
					break;
				}
				// Cj <- t0
				int outlierTransactionIndex = m_outliers.back().first;
				transaction = m_dataset.row(outlierTransactionIndex);
				// Mj <- Mj <- ti
				for (int l = 0; l < m_clustersSums.cols; l++) {
					if ((l % 8) == 0) {
						byte = *(transaction.col((int) l / 8).data);
					}
					m_clustersSums.at<int>(outlierTransactionIndex, l) -= ((int) ((byte >> (7 - (l % 8))) % 2));
				}
				// Nj <- Nj - 1
				m_clustersCounts.col(outlierTransactionIndex) -= 1;
				m_outliers.pop_back();
			}
		}
	}

}

// --------------------------------------------------------------------------

void IncrementalKMeans::save(const std::string& filename) const {
	// TODO Implement this method
}

// --------------------------------------------------------------------------

void IncrementalKMeans::load(const std::string& filename) {
	// TODO Implement this method
}

// --------------------------------------------------------------------------

void IncrementalKMeans::preComputeDistances() {
	cv::Mat nullTransaction = cv::Mat::zeros(1, m_numClusters, cv::DataType<double>::type);
	for (int j = 0; j < m_numClusters; ++j) {
		cv::mulTransposed(nullTransaction - m_centroids.row(j), m_clusterDistancesToNullTransaction.col(j), true);
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::findNearestNeighbor(cv::Mat transaction, int& clusterIndex, double& distanceToCluster) {

	clusterIndex = -1;
	distanceToCluster = std::numeric_limits<double>::max();

	double tempDistanceToCluster;
	uchar byte = 0;
	for (int j = 0; j < m_centroids.rows; ++j) {
		tempDistanceToCluster = m_clusterDistancesToNullTransaction.at<double>(0, j);
		for (int l = 0; l < m_centroids.cols; l++) {
			if ((l % 8) == 0) {
				byte = *(transaction.col((int) l / 8).data);
			}
			int bit = ((int) ((byte >> (7 - (l % 8))) % 2));
			// Compute only differences for non-null dimensions
			if (bit == 1) {
				m_clusterDistances.col(j) += pow(bit - m_centroids.at<double>(j, l), 2) - pow(m_centroids.at<double>(j, l), 2);
			}
		}
		if (tempDistanceToCluster < distanceToCluster) {
			clusterIndex = j;
			distanceToCluster = tempDistanceToCluster;
		}
	}

}

} /* namespace vlr */
