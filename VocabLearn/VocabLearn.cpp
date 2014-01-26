/*
 * VocabLearn.cpp
 *
 *  Created on: Oct 6, 2013
 *      Author: andresf
 */

#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <sys/stat.h>
#include <boost/regex.hpp>

#include <opencv2/core/internal.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabTree.h>

using std::vector;

double mytime;

int main(int argc, char **argv) {

	if (argc < 6 || argc > 6) {
		printf("\nUsage:\n"
				"\tVocabLearn <in.training_images.list> <in.depth>"
				" <in.branch.factor> <in.restarts>"
				" <out.tree>\n\n");
		return EXIT_FAILURE;
	}

	std::string list_in = argv[1];
	int depth = atoi(argv[2]);
	int branchFactor = atoi(argv[3]);
	int restarts = atoi(argv[4]);
	std::string tree_out = argv[5];

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(tree_out, expression) == false) {
		fprintf(stderr,
				"Output tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1: read list of descriptors files to build the tree
	printf("-- Loading list of descriptors files\n");
	std::vector<std::string> descriptorsFilenames;
	FileUtils::loadList(list_in, descriptorsFilenames);
	printf("   Loaded, got [%lu] entries\n", descriptorsFilenames.size());

	// Step 3: build tree
	printf("-- Initializing dynamic descriptors matrix\n");
	vlr::DynamicMat dataset(descriptorsFilenames);
	printf("   Initialized, got [%d] descriptors\n", dataset.rows);

	// Cluster descriptors using Vocabulary Tree
	bfeat::VocabTreeParams params;
	params["branching"] = branchFactor;
	params["iterations"] = restarts;
	params["depth"] = depth;

	cv::Ptr<bfeat::VocabTreeBase> tree;

	printf("-- Descriptor type is [%s]\n",
			dataset.type() == CV_8U ? "binary" : "non-binary");

	if (dataset.type() == CV_8U) {
		tree = new bfeat::VocabTreeBin(dataset, params);
	} else {
		tree = new bfeat::VocabTreeReal(dataset, params);
	}

	printf(
			"-- Building vocabulary tree from [%d] feature vectors, branch factor [%d], max iterations [%d], depth [%d], centers initialization algorithm [%s]\n",
			dataset.rows, params["branching"].cast<int>(),
			params["iterations"].cast<int>(), params["depth"].cast<int>(),
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_RANDOM ? "random" :
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_GONZALES ? "gonzalez" :
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_KMEANSPP ?
					"k-means++" : "unknown");

	mytime = cv::getTickCount();
	tree->build();
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf(
			"   Vocabulary created from [%d] descriptors in [%lf] ms with [%lu] words\n",
			dataset.rows, mytime, tree->size());

	printf("-- Saving tree to [%s]\n", tree_out.c_str());

	mytime = cv::getTickCount();
	tree->save(tree_out);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Tree saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}
