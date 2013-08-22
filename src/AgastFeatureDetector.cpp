/*
 * AgastFeatureDetector.cpp
 *
 *  Created on: Aug 20, 2013
 *      Author: andresf
 */

#include <AgastFeatureDetector.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>

#include <agast5_8.h>
#include <agast7_12s.h>
#include <agast7_12d.h>
#include <oast9_16.h>

#include <AstDetector.h>

namespace cv {

KeyPoint cvPointToKeyPoint(struct CvPoint p) {
	cv::KeyPoint k;
	k.pt.x = p.x;
	k.pt.y = p.y;
	return k;
}

void AGAST(const Mat& _img, std::vector<KeyPoint>& keypoints,
		const int& threshold, const bool& nonmaxsuppression, const int& type) {

	// Obtain detector instance
	AstDetector* detector;

	switch (type) {
	case AgastFeatureDetector::TYPE_OAST9_16:
		detector = new OastDetector9_16(_img.cols, _img.rows, threshold);
		break;
	case AgastFeatureDetector::TYPE_AGAST7_12d:
		detector = new AgastDetector7_12d(_img.cols, _img.rows, threshold);
		break;
	case AgastFeatureDetector::TYPE_AGAST7_12s:
		detector = new AgastDetector7_12s(_img.cols, _img.rows, threshold);
		break;
	case AgastFeatureDetector::TYPE_AGAST5_8:
		detector = new AgastDetector5_8(_img.cols, _img.rows, threshold);
		break;
	default:
		detector = new OastDetector9_16(_img.cols, _img.rows, threshold);
		break;
	}

	// Converting color image to grayscale
	Mat grayImage = _img;
	if (_img.type() != CV_8U)
		cvtColor(_img, grayImage, COLOR_BGR2GRAY);

	// Detect keypoints
	detector->processImage(grayImage.data);
	vector<CvPoint> corners;
	if (nonmaxsuppression == true) {
		corners = detector->get_corners_nms();
	} else {
		corners = detector->get_corners_all();
	}
	vector<int> scores = detector->get_scores();

	// Transform vector of CvPoint into vector of cv::KeyPoint
	// NB: only position information is added because AGAST is not rotation neither scale invariant

	//	std::transform(corners_all.begin(), corners_all.end(), keypoints.begin(), cvPointToKeyPoint);

	for (int i = 0; i < (int) corners.size(); i++) {
		CvPoint corner = corners[i];
		KeyPoint k = cvPointToKeyPoint(corner);
		k.response = scores[i];
		k.size = 7.f;
		keypoints.push_back(k);
	}

}

void AgastFeatureDetector::operator ()(const Mat& image,
		std::vector<KeyPoint>& keypoints) const {
	AGAST(image, keypoints, this->threshold, this->nonmaxsuppression,
			this->type);
}

void AgastFeatureDetector::detectImpl(const Mat& image,
		std::vector<KeyPoint>& keypoints, const Mat& mask) const {
	(*this)(image, keypoints);
}

}
