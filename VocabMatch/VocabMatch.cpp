/*
 * VocabMatch.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#include <stdlib.h>
#include <VocabTree.h>

int main(int argc, char **argv) {

	cvflann::VocabTree tree;
	cv::Mat descriptors;

	int normType = cv::NORM_L1;

	std::vector<cv::Mat> images;

	// Step 4/4: Quantize testing/query data and obtain BoW representation, then score them against DB bow vectors
	std::vector<cv::Mat> queriesDescriptors;
	queriesDescriptors.push_back(descriptors);

	printf("-- Scoring [%lu] query images against [%lu] DB images using [%s]\n",
			queriesDescriptors.size(), images.size(),
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");

	for (size_t i = 0; i < queriesDescriptors.size(); i++) {
		cv::Mat scores;
		tree.scoreQuery(queriesDescriptors[i], scores, 1, cv::NORM_L1);

		for (size_t j = 0; (int) j < scores.rows; j++) {
			printf(
					"   Match score between [%lu] query image and [%lu] DB image: %f\n",
					i, j, scores.at<float>(0, j));
		}
	}

	return EXIT_SUCCESS;
}

