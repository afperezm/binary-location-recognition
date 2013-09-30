/*
 * BOWKmajorityTrainer.h
 *
 *  Created on: Sep 26, 2013
 *      Author: andresf
 */

#ifndef BOWKMAJORITYTRAINER_H_
#define BOWKMAJORITYTRAINER_H_

#include <opencv2/features2d/features2d.hpp>

namespace cv {

class BOWKmajorityTrainer: public BOWTrainer {

public:
	BOWKmajorityTrainer(int clusterCount, int maxIterations,
			cvflann::flann_centers_init_t centers_init =
					cvflann::FLANN_CENTERS_RANDOM);
	virtual ~BOWKmajorityTrainer();
	virtual Mat cluster() const;
	virtual Mat cluster(const Mat& descriptors) const;

protected:
	int m_clusterCount;
	int m_maxIterations;
	cvflann::flann_centers_init_t m_centers_init;
};

} /* namespace cv */
#endif /* BOWKMAJORITYTRAINER_H_ */
