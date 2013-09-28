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
	BOWKmajorityTrainer(int clusterCount, int _maxIterations, int flags =
			KMEANS_PP_CENTERS);
	virtual ~BOWKmajorityTrainer();
	virtual Mat cluster() const;
	virtual Mat cluster(const Mat& descriptors) const;

protected:
	int m_clusterCount;
	int m_maxIterations;
	int m_flags;
};

} /* namespace cv */
#endif /* BOWKMAJORITYTRAINER_H_ */
