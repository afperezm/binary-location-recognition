/*
 * IncrementalKMeans.cpp
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#include <IncrementalKMeans.hpp>
#include <opencv2/core/core.hpp>

namespace vlr {

static const int MAX_OUTLIERS = 10;

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
	m_clusterDistancesToNullTransaction.create(1, m_numClusters, cv::DataType<double>::type);
	m_outliers.resize(m_numClusters);

}

// --------------------------------------------------------------------------

IncrementalKMeans::~IncrementalKMeans() {
}

// --------------------------------------------------------------------------

void IncrementalKMeans::build() {

	// Mean-based initialization
	initCentroids();
	// Compute distances between clusters centers and the null transaction
	preComputeDistances();
	// Nj <- 0
	// Mj <- 0
	// Wj <- 1/k
	initClustersCounters();

	double L = sqrt(m_numDatapoints);
	int clusterIndex;
	double distanceToCluster;

	for (int i = 0; i < m_numDatapoints; ++i) {
		cv::Mat transaction = m_dataset.row(i);
		// Cluster assignment
		// j = NN(ti)
		findNearestNeighbor(transaction, clusterIndex, distanceToCluster);
		// Insert transaction in the list of outliers
		bool isOutlier = insertOutlier(i, clusterIndex, distanceToCluster);
		// If the transaction is not an outlier then assign it to the jth cluster
		if (isOutlier) {
			// If the transaction is an outlier and is the farthest one on the jth cluster
			// then pop and assign the nearest outlier on the jth cluster
			if (m_outliers.at(clusterIndex).size() > MAX_OUTLIERS) {
				transaction = m_dataset.row(m_outliers.at(clusterIndex).back().first);
				m_outliers.at(clusterIndex).pop_back();
				// Mj <- Mj + ti
				sparseSum(transaction, clusterIndex);
				// Nj <- Nj + 1
				m_clustersCounts.col(clusterIndex) += 1;
			}
		} else {
			// Mj <- Mj + ti
			sparseSum(transaction, clusterIndex);
			// Nj <- Nj + 1
			m_clustersCounts.col(clusterIndex) += 1;
		}

		// Update clusters centers every L times
		if (i % ((int) (m_numDatapoints / L)) == 0) {
			// Re-compute clusters centers
			computeCentroids(i);
			// Compute distances between clusters centers and the null transaction
			preComputeDistances();
			// Re-seeding
			handleEmptyClusters();
		}
	}

}

// --------------------------------------------------------------------------

void IncrementalKMeans::save(const std::string& filename) const {

	if (m_centroids.empty()) {
		throw std::runtime_error("[IncrementalKMeans::save] Vocabulary is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[IncrementalKMeans::save] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "type" << "IKM";
	fs << "C" << m_centroids;
	fs << "R" << m_clustersVariances;
	fs << "W" << m_clustersWeights;

	fs.release();

}

// --------------------------------------------------------------------------

void IncrementalKMeans::load(const std::string& filename) {
	// TODO Implement this method
}

// --------------------------------------------------------------------------

void IncrementalKMeans::initCentroids() {
	for (int j = 0; j < m_numClusters; ++j) {
		// Cj <- miu +/-sigma*r/d
		double r = (double) rand() / RAND_MAX;
		if (rand() % 2 == 0) {
			m_centroids.row(j) = (m_miu + m_sigma * r / (m_dim * 8));
		} else {
			m_centroids.row(j) = (m_miu - m_sigma * r / (m_dim * 8));
		}
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::preComputeDistances() {
	cv::Mat nullTransaction = cv::Mat::zeros(1, m_dim * 8, cv::DataType<double>::type);
	for (int j = 0; j < m_numClusters; ++j) {
		cv::mulTransposed(nullTransaction - m_centroids.row(j), m_clusterDistancesToNullTransaction.col(j), false);
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::initClustersCounters() {
	m_clustersCounts = cv::Mat::zeros(1, m_numClusters, cv::DataType<int>::type);
	m_clustersSums = cv::Mat::zeros(m_numClusters, m_dim * 8, cv::DataType<int>::type);
	m_clustersWeights = cv::Mat::ones(1, m_numClusters, cv::DataType<double>::type) / m_numClusters;
}

// --------------------------------------------------------------------------

void IncrementalKMeans::findNearestNeighbor(cv::Mat transaction, int& clusterIndex, double& distanceToCluster) {

	clusterIndex = 0;
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
				tempDistanceToCluster += pow(bit - m_centroids.at<double>(j, l), 2) - pow(m_centroids.at<double>(j, l), 2);
			}
		}
		if (tempDistanceToCluster < distanceToCluster) {
			clusterIndex = j;
			distanceToCluster = tempDistanceToCluster;
		}
	}

}

// --------------------------------------------------------------------------

bool IncrementalKMeans::insertOutlier(const int& transactionIndex, const int& clusterIndex, const double& distanceToCluster) {
	// Retrieve list of outliers for the given cluster index
//	std::vector<std::pair<int, double>> clusterOutliers = m_outliers.at(clusterIndex);
	// Insert item into the list of outliers for the given cluster index
	std::pair<int, double> item(transactionIndex, distanceToCluster);
	// Limit size of outliers list
	if(m_outliers.at(clusterIndex).size() < MAX_OUTLIERS) {
		std::vector<std::pair<int, double>>::iterator position = std::upper_bound(m_outliers.at(clusterIndex).begin(), m_outliers.at(clusterIndex).end(), item, [](const std::pair<int, double>& lhs, const std::pair<int, double>& rhs) {return lhs.second >= rhs.second;});
		m_outliers.at(clusterIndex).insert(position, item);
		return true;
	} else {
		if (item.second >= m_outliers.at(clusterIndex).front().second) {
			m_outliers.at(clusterIndex).insert(m_outliers.at(clusterIndex).begin(), item);
			return true;
		} else {
			return false;
		}
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::sparseSum(cv::Mat transaction, const int& rowIndex) {
	uchar byte = 0;
	for (int l = 0; l < m_clustersSums.cols; l++) {
		if ((l % 8) == 0) {
			byte = *(transaction.col((int) l / 8).data);
		}
		m_clustersSums.at<int>(rowIndex, l) += ((int) ((byte >> (7 - (l % 8))) % 2));
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::sparseSubtraction(cv::Mat transaction, const int& rowIndex) {
	uchar byte = 0;
	for (int l = 0; l < m_clustersSums.cols; l++) {
		if ((l % 8) == 0) {
			byte = *(transaction.col((int) l / 8).data);
		}
		m_clustersSums.at<int>(rowIndex, l) -= ((int) ((byte >> (7 - (l % 8))) % 2));
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::computeCentroids(const int& i) {
	for (int j = 0; j < m_numClusters; ++j) {
		// Cj <- Mj/Nj
		m_clustersSums.row(j).convertTo(m_centroids.row(j), cv::DataType<double>::type);
		m_centroids.row(j) = m_centroids.row(j) / ((double) m_clustersCounts.at<int>(0, j));
		// Rj <- Cj - diag(Cj*Cj')
		cv::Mat clusterVariance(m_dim * 8, m_dim * 8, cv::DataType<double>::type);
		cv::mulTransposed(m_centroids.row(j), clusterVariance, true);
		m_clustersVariances.row(j) = m_centroids.row(j) - clusterVariance.diag(0).t();
		// Wj <- Nj/i
		m_clustersWeights.col(j) = ((double) m_clustersCounts.at<int>(0, j)) / ((double) i);
	}
}

// --------------------------------------------------------------------------

void IncrementalKMeans::handleEmptyClusters() {
	for (unsigned int j = 0; j < m_numClusters; ++j) {
		bool allEmpty = true;
		for (unsigned int i = 0; i < m_outliers.size(); ++i) {
			allEmpty = allEmpty && m_outliers.at(i).empty();
		}
		if (allEmpty) {
			break;
		}
		// if Wj = 0 then Cj <- to
		if (m_clustersWeights.at<double>(0, j) != 0) {
			continue;
		}
		// Get an outlier transaction assigned to a cluster different than the jth cluster
		int outlierTransactionIndex;
		for (unsigned int i = 0; i < m_outliers.size(); ++i) {
			if (i !=j && !m_outliers.at(i).empty()) {
				outlierTransactionIndex = m_outliers.at(i).front().first;
				m_outliers.at(i).pop_back();
				break;
			}
		}
		cv::Mat outlierTransaction = m_dataset.row(outlierTransactionIndex);
		// Assign outlier transaction to the jth cluster
		// Mj <- Mj + ti
		sparseSum(outlierTransaction, j);
		// Nj <- Nj + 1
		m_clustersCounts.col(j) += 1;
	}
}

} /* namespace vlr */
