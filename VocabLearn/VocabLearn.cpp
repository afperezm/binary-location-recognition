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
#include <iostream>
#include <fstream>
#include <ctime>
#include <sys/stat.h>

#include <opencv2/core/internal.hpp>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

// KMajority
#include <Clustering.h>
#include <BOWKmajorityTrainer.h>

// VocabTree
#include <VocabTree.h>

// DBoW2
//#include <DBoW2.h>
//#include <DUtilsCV.h>
//#include <DVision.h>

#include <FileUtils.hpp>

using cv::Mat;
using std::vector;

double mytime;

int main(int argc, char **argv) {

	// Initiating non-free module, it's necessary for using SIFT and SURF
//	cv::initModule_nonfree();

//	printf("-- Loading image [%s]\n", argv[1]);

//	mytime = (double) cv::getTickCount();
//	Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;

//	if (img_1.empty()) {
//		fprintf(stderr, "-- Error reading image [%s]\n", argv[1]);
//	}
//	printf("-- Image loaded in [%lf] ms\n", mytime);

// Step 1/5: detect keypoints using FAST or AGAST
//	std::vector<cv::KeyPoint> keypoints;
//	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
//			"AGAST");
//	printParams(detector);

//	mytime = cv::getTickCount();
//	detector->detect(img_1, keypoints);
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;
//	printf("-- Detected [%zu] keypoints in [%lf] ms\n", keypoints.size(),
//			mytime);

// Step 2/5: extract descriptors using BRIEF or DBRIEF
//	cv::Ptr<cv::DescriptorExtractor> extractor =
//			cv::DescriptorExtractor::create("BRIEF");

//	Mat descriptors;

//	mytime = cv::getTickCount();
//	extractor->compute(img_1, keypoints, descriptors);
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;

// Notice that number of keypoints might be reduced due to border effect
//	printf(
//			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
//			descriptors.rows, descriptors.cols,
//			descriptors.type() == CV_8U ? "binary" : "real-valued", mytime);

// Step 3/5: Show keypoints

//	cv::drawKeypoints(img_1, keypoints, img_1, cv::Scalar::all(-1));
//	cv::namedWindow("Image keypoints", CV_WINDOW_NORMAL);
//	cv::imshow("Image keypoints", img_1);

//	cv::waitKey(0);

// Step 4/5: save descriptors into a file for later use
//	FileUtils::save("test.xml.gz", keypoints, descriptors);

// Step 5/5: Cluster descriptors using binary vocabulary tree aided by k-means

// Transform descriptors to a suitable structure for DBoW2
//	printf("-- Transforming descriptors to a suitable structure for DBoW2\n");
//	vector<vector<DVision::BRIEF::bitset> > features;
//	features.resize(1);
//
//	features[0].resize(descriptors.rows);
//
//	for (unsigned int i = 0; (int) i < descriptors.rows * descriptors.cols;
//			i++) {
//		int row = (int) i / descriptors.cols;
//		int col = (int) i % descriptors.cols;
//		if (col == 0) {
//			(features[0])[row].resize(descriptors.cols * 8);
//			(features[0])[row].reset();
//		}
//		unsigned char byte = descriptors.at<uchar>(row, col);
//		for (int i = 0 + (descriptors.cols - 1 - col) * 8;
//				i <= 7 + (descriptors.cols - 1 - col) * 8; i++) {
//			((features[0])[row])[i] = byte & 1;
//			byte >>= 1;
//		}
//	}
//
//	printf("   Obtained [%zu] vectors\n", features.size());
//
//	const int k = 6;
//	const int L = 3;
//	const DBoW2::WeightingType weight = DBoW2::TF_IDF;
//	const DBoW2::ScoringType score = DBoW2::L1_NORM;
//
//	BriefVocabulary voc(k, L, weight, score);
//	mytime = cv::getTickCount();
//	voc.create(features);
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;
//	printf(
//			"-- Vocabulary created in [%lf] ms with NWORDS=%u BRANCHING=%d DEPTH=%d SCORING=%s WEIGHTING=%s\n",
//			mytime, voc.size(), k, L,
//			weight == DBoW2::TF_IDF ? "tf-idf" : weight == DBoW2::TF ? "tf" :
//			weight == DBoW2::IDF ? "idf" :
//			weight == DBoW2::BINARY ? "binary" : "unknown",
//			score == DBoW2::L1_NORM ? "L1-norm" :
//			score == DBoW2::L2_NORM ? "L2-norm" :
//			score == DBoW2::CHI_SQUARE ? "Chi square distance" :
//			score == DBoW2::KL ? "KL-divergence" :
//			score == DBoW2::BHATTACHARYYA ? "Bhattacharyya coefficient" :
//			score == DBoW2::DOT_PRODUCT ? "Dot product" : "unknown");

	if (argc != 6) {
		printf("\nUsage:\n"
				"\t%s <list.in> <depth>"
				" <branching_factor> <restarts>"
				" <tree.out>\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char *list_in = argv[1];
	int depth = atoi(argv[2]);
	int branchFactor = atoi(argv[3]);
	int restarts = atoi(argv[4]);
	const char *tree_out = argv[5];

	printf("-- Building tree with depth: %d, branching factor: %d, "
			"and restarts: %d\n", depth, branchFactor, restarts);

	// Step 1: read list of key files that shall be used to build the tree
	std::vector<std::string> keysFilenames;
	std::ifstream keysList(list_in, std::fstream::in);

	if (keysList.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading", list_in);
	}

	// Loading file names in list into a vector
	std::string line;
	while (getline(keysList, line)) {
		struct stat buffer;
		// Checking if file exist, if not print error and exit
		if (stat(line.c_str(), &buffer) == 0) {
			keysFilenames.push_back(line);
		} else {
			fprintf(stderr, "Found invalid keypoints file [%s]", line.c_str());
			return EXIT_FAILURE;
		}
	}
	// Close file
	keysList.close();

	// Step 2: read key files
	std::vector<cv::KeyPoint> keypoints;
	cv::Mat descriptors, mergedDescriptors;

	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		keypoints.clear();
		descriptors = cv::Mat();
		FileUtils::loadFeatures(keyFileName, keypoints, descriptors);
		// Check that keypoints and descriptors have same length
		CV_Assert((int )keypoints.size() == descriptors.rows);
//		tree.add(descriptors);
	}

	// Step 3: build tree

//	// Cluster descriptors using Vocabulary Tree
//	cvflann::VocabTreeParams params;
//	// TODO Add method for composing descriptors big matrix from the list of descriptors references
//	cvflann::VocabTree tree(descriptors, params);
//
//	printf(
//			"-- Creating vocabulary tree using [%d] feature vectors, branch factor [%d], max iterations [%d], depth [%d], centers init algorithm [%s]\n",
//			descriptors.rows, params["branching"].cast<int>(),
//			params["iterations"].cast<int>(), params["depth"].cast<int>(),
//			params["centers_init"].cast<cvflann::flann_centers_init_t>()
//					== cvflann::FLANN_CENTERS_RANDOM ? "random" :
//			params["centers_init"].cast<cvflann::flann_centers_init_t>()
//					== cvflann::FLANN_CENTERS_GONZALES ? "gonzalez" :
//			params["centers_init"].cast<cvflann::flann_centers_init_t>()
//					== cvflann::FLANN_CENTERS_KMEANSPP ?
//					"k-means++" : "unknown");
//
//	mytime = cv::getTickCount();
//	tree.build();
//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
//			* 1000;
//	printf("   Vocabulary created in [%lf] ms with [%lu] words\n", mytime,
//			tree.size());
//
//	std::string treeOut = "tree.yaml.gz";
//	printf("   Saving tree to [%s]\n", treeOut.c_str());
//	tree.save(treeOut);

	delete list_in;
	delete tree_out;

	return EXIT_SUCCESS;
}
