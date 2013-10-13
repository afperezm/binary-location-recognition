/*
 * features2d_extensions.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <opencv2/extensions/features2d.hpp>
#include <AgastFeatureDetector.h>
#include <DBriefDescriptorExtractor.h>

namespace cv {

CV_INIT_ALGORITHM(AgastFeatureDetector, "Feature2D.AGAST",
		obj.info()->addParam(obj, "threshold", obj.threshold); obj.info()->addParam(obj, "nonmaxsuppression", obj.nonmaxsuppression); obj.info()->addParam(obj, "type", obj.type))
;

CV_INIT_ALGORITHM(DBriefDescriptorExtractor, "Feature2D.DBRIEF", obj.info())
;

bool initModule_features2d_extensions() {
	AgastFeatureDetector_info_auto.name();
	DBriefDescriptorExtractor_info_auto.name();

	cv::Ptr<AgastFeatureDetector> agast = new AgastFeatureDetector();
	cv::Ptr<DBriefDescriptorExtractor> dbrief = new DBriefDescriptorExtractor();

	return agast->info() != 0 && dbrief->info() != 0;
}

}
