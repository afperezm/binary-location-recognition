/*
 * DBriefDescriptorExtractor.cpp
 *
 *  Created on: Aug 22, 2013
 *      Author: andresf
 */

#include <DBriefDescriptorExtractor.h>

namespace cv {

DBriefDescriptorExtractor::DBriefDescriptorExtractor() {
	// TODO Auto-generated constructor stub

}

DBriefDescriptorExtractor::~DBriefDescriptorExtractor() {
	// TODO Auto-generated destructor stub
}

int DBriefDescriptorExtractor::descriptorSize() const {
	return -1;
}

int DBriefDescriptorExtractor::descriptorType() const {
	return -1;
}

void DBriefDescriptorExtractor::computeImpl(const Mat& image,
		std::vector<KeyPoint>& keypoints, Mat& descriptors) const{

}

} /* namespace cv */
