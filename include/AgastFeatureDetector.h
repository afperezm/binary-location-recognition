/*
 * AgastFeatureDetector.h
 *
 *  Created on: Aug 20, 2013
 *      Author: andresf
 */

#ifndef AGASTFEATUREDETECTOR_H_
#define AGASTFEATUREDETECTOR_H_

#include <opencv2/features2d/features2d.hpp>

#include <AstDetector.h>

#include <stdio.h>

namespace cv {

class CV_EXPORTS_W AgastFeatureDetector: public FeatureDetector {
public:

	enum AST_PATTERN {
		TYPE_OAST9_16, TYPE_AGAST7_12d, TYPE_AGAST7_12s, TYPE_AGAST5_8
	};

	//! the full constructor
	AgastFeatureDetector() :
			threshold(10), type(TYPE_OAST9_16), nonmaxsuppression(true) {
		;
	}
	AgastFeatureDetector(int _threshold, int _type, bool _nonmaxsuppression) :
			threshold(_threshold), type(_type), nonmaxsuppression(
					_nonmaxsuppression) {
		;
	}
	~AgastFeatureDetector() {
		;
	}

	//! finds the keypoints in the image
	CV_WRAP_AS(detect)
	void operator()(const Mat& image,
			CV_OUT std::vector<KeyPoint>& keypoints) const;

	AlgorithmInfo* info() const;

protected:

	// Detect keypoints in an image set
	void detectImpl(const Mat& image, std::vector<KeyPoint>& keypoints,
			const Mat& mask = Mat()) const;

	int threshold;
	int type;
	bool nonmaxsuppression;

};

} /* namespace cv */

#endif /* AGASTFEATUREDETECTOR_H_ */
