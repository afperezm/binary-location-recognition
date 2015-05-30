/*
 * IncrementalKMeans.cpp
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#include <IncrementalKMeans.h>
#include <opencv2/core/core.hpp>

namespace vlr {

IncrementalKMeans::IncrementalKMeans(cv::Mat data,
		const cvflann::IndexParams& params) :
		m_dataset(data), m_dim(data.cols), m_numDatapoints(data.rows) {

	// Attributes initialization
	m_numClusters = cvflann::get_param<int>(params, "num.clusters");

	// Compute the global data set mean
	m_miu = cv::Mat::zeros(1, m_dim, CV_32F);
	for (int i = 0; i < m_numDatapoints; ++i) {
		m_miu += m_dataset.row(i);
	}
	m_miu /= m_numDatapoints;

	// Compute the global data set standard deviations
	m_sigma = cv::Mat::zeros(1, m_dim, CV_32F);
	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat temp(1, m_dim, CV_32F);
		cv::pow(m_dataset.row(i) - m_miu, 2, temp);
		m_sigma += temp;
	}
	m_sigma /= m_numDatapoints;
	cv::sqrt(m_sigma, m_sigma);

}

// --------------------------------------------------------------------------

IncrementalKMeans::~IncrementalKMeans() {
}

// --------------------------------------------------------------------------

void IncrementalKMeans::build() {

	m_centroids.create(m_numClusters, m_dim, CV_32F);
	m_clustersVariances.create(m_numClusters, m_dim, CV_32F);

	// Mean-based initialization
	srand(time(NULL));
	for (int j = 0; j < m_numClusters; ++j) {
		// Cj <- miu +/-sigma*r/d
		double r = (double) rand() / RAND_MAX;
		if (rand() % 2 == 0) {
			m_centroids.row(j) = (m_miu + m_sigma * r / m_dim);
		} else {
			m_centroids.row(j) = (m_miu - m_sigma * r / m_dim);
		}
	}
	// Nj <- 0
	m_clustersCounts = cv::Mat::zeros(1, m_numClusters, CV_32F);
	// Mj <- 0
	m_clustersSums = cv::Mat::zeros(m_numClusters, m_dim, CV_32F);
	// Wj <- 1/k
	m_clustersWeights = cv::Mat::ones(1, m_numClusters, CV_32F) / m_numClusters;

	double L = sqrt(m_numDatapoints);

	for (int i = 0; i < m_numDatapoints; ++i) {
		// Cluster assignment
		// j = NN(ti)
		int J = findNearestNeighbor(m_dataset.row(i));
		double distance = cv::norm(m_dataset.row(i), m_centroids.row(J));
		// Mj <- Mj + ti
		m_clustersSums.row(J) += m_dataset.row(i);
		// Nj <- Nj + 1
		m_clustersCounts.col(J) += 1;
		std::pair<int, double> item(J, distance);
		// Insert outliers in an ordered manner
		m_outliers.insert(
				std::upper_bound(m_outliers.begin(), m_outliers.end(), item,
						[](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs)
						{	return lhs.second <= rhs.second;}), item);

		if (i % ((int) (m_numDatapoints / L)) == 0) {
			// Re-compute clusters centers
			for (int j = 0; j < m_numClusters; ++j) {
				// Cj <- Mj/Nj
				m_centroids.row(J) = m_clustersSums.row(j)
						/ m_clustersCounts.col(j);
				// Rj <- Cj - diag(Cj*Cj')
				cv::Mat clusterVariance(m_dim, m_dim, CV_32F);
				cv::mulTransposed(m_centroids.row(j), clusterVariance, true);
				m_clustersVariances.row(j) = m_centroids.row(j)
						- clusterVariance.diag(0);
				// Wj <- Nj/i
				m_clustersWeights.col(j) = m_clustersCounts.col(j) / i;
			}
			// Re-seeding
			for (int j = 0; j < m_numClusters; ++j) {
				// Cj <- t0
			}
		}
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::save(const std::string& filename) const {
}

// --------------------------------------------------------------------------

void IncrementalKMeans::load(const std::string& filename) {
}

int IncrementalKMeans::findNearestNeighbor(cv::Mat transaction) {
	int J = -1;
	return J;
}

template<typename T>
typename std::vector<T>::iterator insertSorted(std::vector<T> & vec,
		T const& item) {
	return vec.insert(std::upper_bound(vec.begin(), vec.end(), item), item);
}

} /* namespace vlr */
