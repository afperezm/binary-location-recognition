/*
 * BOWKmajorityTrainer.cpp
 *
 *  Created on: Sep 26, 2013
 *      Author: andresf
 */

#include <BOWKmajorityTrainer.h>
#include <Clustering.h>

namespace cv {

BOWKmajorityTrainer::BOWKmajorityTrainer(int clusterCount, int maxIterations,
		cvflann::flann_centers_init_t centers_init) :
		m_clusterCount(clusterCount), m_maxIterations(maxIterations), m_centers_init(
				centers_init) {
}

BOWKmajorityTrainer::~BOWKmajorityTrainer() {
}

Mat BOWKmajorityTrainer::cluster() const {

	CV_Assert(descriptors.empty() == false);

	// Compute number of rows of matrix containing all training descriptors,
	// that is matrix resulting from the concatenation of the images descriptors
	int descriptorsCount = 0;
	for (size_t i = 0; i < descriptors.size(); i++) {
		descriptorsCount += descriptors[i].rows;
	}

	// Concatenating the images descriptors into a single big matrix
	Mat trainingDescriptors(descriptorsCount, descriptors[0].cols,
			descriptors[0].type());

	for (size_t i = 0, start = 0; i < descriptors.size(); i++) {
		Mat submut = trainingDescriptors.rowRange((int) start,
				(int) (start + descriptors[i].rows));
		descriptors[i].copyTo(submut);
		start += descriptors[i].rows;
	}

	return cluster(trainingDescriptors);
}

Mat BOWKmajorityTrainer::cluster(const Mat& descriptors) const {

	Mat vocabulary;

	std::vector<int> labels;

	clustering::kmajority(m_clusterCount, m_maxIterations, descriptors,
			vocabulary, labels, m_centers_init);

	return vocabulary;
}

} /* namespace cv */
