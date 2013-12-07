//============================================================================
// Name        : MediaEval-PlacingTask.cpp
// Author      : Andrés Pérez
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <fstream>
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

	if (argc < 6 || argc > 7) {
		printf("\nUsage:\n"
				"\tVocabLearn <in.train.imgs.list> <in.depth>"
				" <in.branch.factor> <in.restarts>"
				" <out.tree> [in.type.binary:1]\n\n");
		return EXIT_FAILURE;
	}

	const char *list_in = argv[1];
	int depth = atoi(argv[2]);
	int branchFactor = atoi(argv[3]);
	int restarts = atoi(argv[4]);
	const char *tree_out = argv[5];
	bool isDescriptorBinary = true;

	if (argc >= 7) {
		isDescriptorBinary = atoi(argv[6]);
	}

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(std::string(tree_out), expression) == false) {
		fprintf(stderr,
				"Output tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1: read list of key files that shall be used to build the tree
	std::vector<std::string> descsFilenames;
	std::ifstream descList(list_in, std::fstream::in);

	if (descList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	std::string line;
	while (getline(descList, line)) {
		struct stat buffer;
		// Checking if file exist, if not print error and exit
		if (stat(line.c_str(), &buffer) == 0) {
			descsFilenames.push_back(line);
		} else {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
					line.c_str());
			return EXIT_FAILURE;
		}
	}
	// Close file
	descList.close();

	// Step 2: read key files
	printf("-- Reading keypoint files from [%s]\n", list_in);

	// Step 3: build tree
	DynamicMat mergedDescriptors(descsFilenames);

	// Cluster descriptors using Vocabulary Tree
	bfeat::VocabTreeParams params;
	params["branching"] = branchFactor;
	params["iterations"] = restarts;
	params["depth"] = depth;

	cv::Ptr<bfeat::VocabTreeBase> tree;

	if ((mergedDescriptors.type() == CV_8U) != isDescriptorBinary) {
		fprintf(stderr,
				"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
				isDescriptorBinary == true ? "binary" : "non-binary",
				mergedDescriptors.type() == CV_8U ? "binary" : "real");
		return EXIT_FAILURE;
	}

	if (isDescriptorBinary == true) {
		tree = new bfeat::VocabTree<uchar, cv::Hamming>(mergedDescriptors,
				params);
	} else {
		tree = new bfeat::VocabTree<float, cv::L2<float> >(mergedDescriptors,
				params);
	}

	printf(
			"-- Building vocabulary tree from [%d] feature vectors, branch factor [%d], max iterations [%d], depth [%d], centers initialization algorithm [%s]\n",
			mergedDescriptors.rows, params["branching"].cast<int>(),
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
			mergedDescriptors.cols, mytime, tree->size());

	printf("-- Saving tree to [%s]\n", tree_out);

	mytime = cv::getTickCount();
	tree->save(tree_out);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Tree saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}
