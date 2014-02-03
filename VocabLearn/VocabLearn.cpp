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
		printf(
				"\nUsage:\n"
						"\tVocabLearn <in.training.images.list> <in.vocab.type> <out.vocab>"
						" <in.depth> <in.branch.factor> <in.restarts>\n\n"
						"Vocabulary type:\n"
						"\tHKM: Hierarchical K-Means\n"
						"\tHKMaj: Hierarchical K-Majority\n"
						"\tAKMaj: Approximate K-Majority\n"
						"\tAKM: Approximate K-Means\n\n");
		return EXIT_FAILURE;
	}

	std::string in_train_list = argv[1];
	std::string in_vocab_type = argv[2];
	std::string out_vocab = argv[3];
	// TODO Generalize arguments to any vocabulary
	int in_depth = atoi(argv[4]);
	int in_branchFactor = atoi(argv[5]);
	int in_restarts = atoi(argv[6]);

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(out_vocab, expression) == false) {
		fprintf(stderr,
				"Output tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1: read list of descriptors files to build the tree
	printf("-- Loading list of descriptors files\n");
	std::vector<std::string> descriptorsFilenames;
	FileUtils::loadList(in_train_list, descriptorsFilenames);
	printf("   Loaded, got [%lu] entries\n", descriptorsFilenames.size());

	// Step 3: build tree
	printf("-- Initializing dynamic descriptors matrix\n");
	vlr::Mat dataset(descriptorsFilenames);
	printf("   Initialized, got [%d] descriptors\n", dataset.rows);

	// Cluster descriptors using Vocabulary Tree
	vlr::VocabTreeParams params;
	params["branching"] = in_branchFactor;
	params["iterations"] = in_restarts;
	params["depth"] = in_depth;

	cv::Ptr<vlr::VocabTreeBase> tree;

	printf("-- Descriptor type is [%s]\n",
			dataset.type() == CV_8U ? "binary" : "non-binary");

	if (dataset.type() == CV_8U) {
		tree = new vlr::VocabTreeBin(dataset, params);
	} else {
		tree = new vlr::VocabTreeReal(dataset, params);
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

	printf("-- Saving tree to [%s]\n", out_vocab.c_str());

	mytime = cv::getTickCount();
	tree->save(out_vocab);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Tree saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}
