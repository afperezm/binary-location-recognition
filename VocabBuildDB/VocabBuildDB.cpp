/*
 * VocabBuildDB.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>

#include <VocabTree.h>

#include <FileUtils.hpp>

double mytime;

int main(int argc, char **argv) {

	if (argc < 4 || argc > 7) {
		printf("\nUsage:\n"
				"\t%s <in.list> <in.tree> <out.tree> [in.type.binary:1]"
				" [in.use.tfidf:1] [in.normalize:1]\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	bool use_tfidf = true;
	bool normalize = true;

	char *list_in = argv[1];
	char *tree_in = argv[2];
	char *tree_out = argv[3];
	bool isDescriptorBinary = true;

	if (argc >= 5) {
		isDescriptorBinary = atoi(argv[4]);
	}

	if (argc >= 6) {
		use_tfidf = atoi(argv[5]);
	}

	if (argc >= 7) {
		normalize = atoi(argv[6]);
	}

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(std::string(tree_in), expression) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	if (boost::regex_match(std::string(tree_out), expression) == false) {
		fprintf(stderr,
				"Output tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: read list of key files that shall be used to build the tree
	std::vector<std::string> keysFilenames;
	std::ifstream keysList(list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n", list_in);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	std::string line;
	while (getline(keysList, line)) {
		// Checking that file exists, if not print error and exit
		struct stat buffer;
		if (stat(line.c_str(), &buffer) != 0) {
			fprintf(stderr, "Keypoints file [%s] doesn't exist\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		// Checking file extension to be compressed yaml or xml
		if (boost::regex_match(line, expression) == false) {
			fprintf(stderr,
					"Keypoints file [%s] must have the extension .yaml.gz or .xml.gz\n",
					line.c_str());
			return EXIT_FAILURE;
		}

		keysFilenames.push_back(line);
	}
	// Close file
	keysList.close();

	printf("-- Building DB using [%lu] images\n", keysFilenames.size());

	cv::Ptr<cvflann::VocabTreeBase> tree;

	if (isDescriptorBinary == true) {
		tree = new cvflann::VocabTree<uchar, cv::Hamming>();
	} else {
		tree = new cvflann::VocabTree<float, cv::L2<float> >();
	}

	printf("-- Reading tree from [%s]\n", tree_in);

	mytime = cv::getTickCount();
	tree->load(std::string(tree_in));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	// Step 2/4: Quantize training data (several image descriptor matrices)
	printf("-- Creating vocabulary database with [%lu] images\n",
			keysFilenames.size());
	tree->clearDatabase();
	printf("   Clearing Inverted Files\n");

	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors;
	uint imgIdx = 0;

	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		imgKeypoints.clear();
		imgDescriptors = cv::Mat();
		FileUtils::loadFeatures(keyFileName, imgKeypoints, imgDescriptors);

		// Check type of descriptors
		if ((imgDescriptors.type() == CV_8U) != isDescriptorBinary) {
			fprintf(stderr,
					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
					isDescriptorBinary == true ? "binary" : "non-binary",
					imgDescriptors.type() == CV_8U ? "binary" : "real");
			return EXIT_FAILURE;
		}

		if (imgDescriptors.rows != 0) {
			printf("   Adding image [%u] to database\n", imgIdx);
			try {
				tree->addImageToDatabase(imgIdx, imgDescriptors);
			} catch (const std::runtime_error& error) {
				fprintf(stderr, "%s\n", error.what());
				return EXIT_FAILURE;
			}
		}

		imgIdx++;
	}

	imgDescriptors.release();

	CV_Assert(keysFilenames.size() == imgIdx);

	printf("   Added [%u] images\n", imgIdx);

	// Step 3/4: Compute words weights and normalize DB

	cvflann::WeightingType weightingScheme = cvflann::BINARY;
	if (use_tfidf) {
		weightingScheme = cvflann::TF_IDF;
	} else {
		weightingScheme = cvflann::BINARY;
	}

	printf("-- Computing words weights using a [%s] weighting scheme\n",
			weightingScheme == cvflann::TF_IDF ? "TF-IDF" :
			weightingScheme == cvflann::BINARY ? "BINARY" : "UNKNOWN");

	tree->computeWordsWeights(weightingScheme);

	printf("-- Applying words weights to the DB BoW vectors counts\n");
	tree->createDatabase();

	int normType = cv::NORM_L1;

	if (normalize == true) {
		printf("-- Normalizing DB BoW vectors using [%s]\n",
				normType == cv::NORM_L1 ? "L1-norm" :
				normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");
		tree->normalizeDatabase(normType);
	}

	printf("-- Saving tree with inverted files and weights to [%s]\n",
			tree_out);

	mytime = cv::getTickCount();
	tree->save(tree_out);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Tree saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}

