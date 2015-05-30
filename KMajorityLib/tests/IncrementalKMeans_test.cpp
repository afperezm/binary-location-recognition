/*
 * IncrementalKMeans_test.cpp
 *
 *  Created on: May 13, 2015
 *      Author: andresf
 */

#include <gtest/gtest.h>

#include <FileUtils.hpp>
#include <IncrementalKMeans.h>

namespace vlr {

TEST(IncrementalKMeans, Instantiation) {

	cv::Mat emptyMat;
	vlr::IncrementalKMeans obj(emptyMat);

}

TEST(IncrementalKMeans, Constructor) {

	cv::Mat imgDescriptors;

	FileUtils::loadDescriptors("sift_0.bin", imgDescriptors);

	/*
	 std::vector<std::string> keysFilenames(1, "sift_0.bin");
	 vlr::Mat data(keysFilenames);
	 */

	vlr::IncrementalKMeans obj(imgDescriptors);

}

} /* namespace vlr */
