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

#include <opencv2/core/internal.hpp>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <Clustering.h>
#include <VocabTree.h>
#include <BOWKmajorityTrainer.h>

// DBoW2
#include <DBoW2.h>
#include <DUtilsCV.h>
#include <DVision.h>

using cv::Mat;
using std::vector;

/**
 *
 * @param keypoints
 */
void printKeypoints(std::vector<cv::KeyPoint>& keypoints);

/**
 *
 *
 * @param descriptors
 */
void printDescriptors(const Mat& descriptors);

/**
 * Function for printing to standard output the parameters of <i>algorithm</i>
 *
 * @param algorithm A pointer to an object of type cv::Algorithm
 */
void printParams(cv::Ptr<cv::Algorithm> algorithm);

/**
 * Saves a set of features (keypoints and descriptors) onto a plain text
 * file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to save the features
 * @param keypoints - The keypoints to be saved
 * @param descriptors - The descriptors to be saved
 */
void save(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints, const Mat& descriptors);

/**
 *
 * @param filename path to the file where the features will be written
 * @param keypoints vector of keypoints
 * @param descriptors matrix of descriptors
 */
void writeFeaturesToFile(const std::string& filename,
		const std::vector<cv::KeyPoint>& keypoints, const Mat& descriptors);

int NumberOfSetBits(int i);

int BinToDec(const cv::Mat& binRow);

double mytime;

int main(int argc, char **argv) {

	if (argc != 2) {
		printf("\n");
		printf("Usage: %s <img1>", argv[0]);
		printf("\n\n");
		return EXIT_FAILURE;
	}

	// Initiating non-free module, it's necessary for using SIFT and SURF
	cv::initModule_nonfree();

	printf("-- Loading image [%s]\n", argv[1]);

	mytime = (double) cv::getTickCount();
	Mat img_1 = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	if (img_1.empty()) {
		fprintf(stderr, "-- Error reading image [%s]\n", argv[1]);
	}
	printf("-- Image loaded in [%lf] ms\n", mytime);

	// Step 1/5: detect keypoints using FAST or AGAST
	std::vector<cv::KeyPoint> keypoints_1;
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
			"AGAST");
	printParams(detector);

	mytime = cv::getTickCount();
	detector->detect(img_1, keypoints_1);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("-- Detected [%zu] keypoints in [%lf] ms\n", keypoints_1.size(),
			mytime);

	// Step 2/5: extract descriptors using BRIEF or DBRIEF
	cv::Ptr<cv::DescriptorExtractor> extractor =
			cv::DescriptorExtractor::create("BRIEF");

	Mat descriptors;

	mytime = cv::getTickCount();
	extractor->compute(img_1, keypoints_1, descriptors);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	// Notice that number of keypoints might be reduced due to border effect
	printf(
			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
			descriptors.rows, descriptors.cols,
			descriptors.type() == CV_8U ? "binary" : "real-valued", mytime);

	// Step 3/5: Show keypoints

	cv::drawKeypoints(img_1, keypoints_1, img_1, cv::Scalar::all(-1));
	cv::namedWindow("Image keypoints", CV_WINDOW_NORMAL);
	cv::imshow("Image keypoints", img_1);

	cv::waitKey(0);

	// Step 4/5: save descriptors into a file for later use
	save("test.xml.gz", keypoints_1, descriptors);
	writeFeaturesToFile("test_descriptors", keypoints_1, descriptors);

	// Step 5/5: Cluster descriptors using binary vocabulary tree aided by k-means

	// Transform descriptors to a suitable structure for DBoW2
	printf("-- Transforming descriptors to a suitable structure for DBoW2\n");
	vector<vector<DVision::BRIEF::bitset> > features;
	features.resize(1);

	features[0].resize(descriptors.rows);

	for (unsigned int i = 0; (int) i < descriptors.rows * descriptors.cols;
			i++) {
		int row = (int) i / descriptors.cols;
		int col = (int) i % descriptors.cols;
		if (col == 0) {
			(features[0])[row].resize(descriptors.cols * 8);
			(features[0])[row].reset();
		}
		unsigned char byte = descriptors.at<uchar>(row, col);
		for (int i = 0 + (descriptors.cols - 1 - col) * 8;
				i <= 7 + (descriptors.cols - 1 - col) * 8; i++) {
			((features[0])[row])[i] = byte & 1;
			byte >>= 1;
		}
	}

	printf("   Obtained [%zu] vectors\n", features.size());

	const int k = 6;
	const int L = 3;
	const DBoW2::WeightingType weight = DBoW2::TF_IDF;
	const DBoW2::ScoringType score = DBoW2::L1_NORM;

	BriefVocabulary voc(k, L, weight, score);
	mytime = cv::getTickCount();
	voc.create(features);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf(
			"-- Vocabulary created in [%lf] ms with NWORDS=%u BRANCHING=%d DEPTH=%d SCORING=%s WEIGHTING=%s\n",
			mytime, voc.size(), k, L,
			weight == DBoW2::TF_IDF ? "tf-idf" : weight == DBoW2::TF ? "tf" :
			weight == DBoW2::IDF ? "idf" :
			weight == DBoW2::BINARY ? "binary" : "unknown",
			score == DBoW2::L1_NORM ? "L1-norm" :
			score == DBoW2::L2_NORM ? "L2-norm" :
			score == DBoW2::CHI_SQUARE ? "Chi square distance" :
			score == DBoW2::KL ? "KL-divergence" :
			score == DBoW2::BHATTACHARYYA ? "Bhattacharyya coefficient" :
			score == DBoW2::DOT_PRODUCT ? "Dot product" : "unknown");

	// Step 5/5: Cluster descriptors using Vocabulary Tree
	cvflann::VocabTreeParams params;
	cvflann::VocabTree tree(descriptors, params);

	// Step 5a/d: Build tree
	printf(
			"-- Creating vocabulary tree using [%d] feature vectors, branch factor [%d], max iterations [%d], depth [%d], centers init algorithm [%s]\n",
			descriptors.rows, params["branching"].cast<int>(),
			params["iterations"].cast<int>(), params["depth"].cast<int>(),
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_RANDOM ? "random" :
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_GONZALES ? "gonzalez" :
			params["centers_init"].cast<cvflann::flann_centers_init_t>()
					== cvflann::FLANN_CENTERS_KMEANSPP ?
					"k-means++" : "unknown");

	mytime = cv::getTickCount();
	tree.build();
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Vocabulary created in [%lf] ms with [%lu] words\n", mytime,
			tree.size());

	std::string treeOut = "tree.yaml.gz";
	printf("   Saving tree to [%s]\n", treeOut.c_str());
	tree.save(treeOut);

	// Step 5b/d: Quantize training data (several image descriptor matrices)
	std::vector<cv::Mat> images;
	images.push_back(descriptors);
	printf("-- Creating vocabulary database with [%lu] images\n",
			images.size());
	tree.clearDatabase();
	printf("   Clearing Inverted Files\n");
	for (size_t imgIdx = 0; imgIdx < images.size(); imgIdx++) {
		printf("   Adding image [%lu] to database\n", imgIdx);
		tree.addImageToDatabase(imgIdx, images[imgIdx]);
	}

	// Step 5c/d: Compute words weights and normalize DB
	const DBoW2::WeightingType weightingScheme = DBoW2::TF_IDF;
	printf("   Computing words weights using a [%s] weighting scheme\n",
			weightingScheme == DBoW2::TF_IDF ? "TF-IDF" :
			weightingScheme == DBoW2::TF ? "TF" :
			weightingScheme == DBoW2::IDF ? "IDF" :
			weightingScheme == DBoW2::BINARY ? "BINARY" : "UNKNOWN");
	tree.computeWordsWeights(descriptors.rows, weightingScheme);
	printf("   Applying words weights to the DB BoW vectors counts\n");
	tree.createDatabase();
	int normType = cv::NORM_L1;
	printf("   Normalizing DB BoW vectors using [%s]\n",
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");
	tree.normalizeDatabase(1, normType);

	std::string dbOut = "db.yaml.gz";
	printf("   Saving DB to [%s]\n", dbOut.c_str());
	tree.save(dbOut);

	// Step 5d/d: Quantize testing/query data and obtain BoW representation, then score them against DB bow vectors
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

void printParams(cv::Ptr<cv::Algorithm> algorithm) {
	std::vector<std::string> parameters;
	algorithm->getParams(parameters);

	for (int i = 0; i < (int) parameters.size(); i++) {
		std::string param = parameters[i];
		int type = algorithm->paramType(param);
		std::string helpText = algorithm->paramHelp(param);
		std::string typeText;

		switch (type) {
		case cv::Param::BOOLEAN:
			typeText = "bool";
			break;
		case cv::Param::INT:
			typeText = "int";
			break;
		case cv::Param::REAL:
			typeText = "real (double)";
			break;
		case cv::Param::STRING:
			typeText = "string";
			break;
		case cv::Param::MAT:
			typeText = "Mat";
			break;
		case cv::Param::ALGORITHM:
			typeText = "Algorithm";
			break;
		case cv::Param::MAT_VECTOR:
			typeText = "Mat vector";
			break;
		}
		std::cout << "Parameter '" << param << "' type=" << typeText << " help="
				<< helpText << std::endl;
	}
}

void printKeypoints(std::vector<cv::KeyPoint>& keypoints) {
	for (cv::KeyPoint k : keypoints) {
		printf(
				"angle=[%f] octave=[%d] response=[%f] size=[%f] x=[%f] y=[%f] class_id=[%d]\n",
				k.angle, k.octave, k.response, k.size, k.pt.x, k.pt.y,
				k.class_id);
	}
}

void printDescriptors(const Mat& descriptors) {
	for (int i = 0; i < descriptors.rows; i++) {
		for (int j = 0; j < descriptors.cols; j++) {
			if (descriptors.type() == CV_8U) {
				bitset<8> byte(descriptors.at<uchar>(i, j));
				printf("%s", byte.to_string().c_str());
			} else {
				printf("%f", (float) descriptors.at<float>(i, j));
			}
		}
//		int decimal = BinToDec(descriptors.row(i));
//		if (descriptors.type() == CV_8U) {
//			printf(" = %ld (%d)", decimal, NumberOfSetBits(decimal));
//		}
		printf("\n");
	}
}

void save(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints, const Mat& descriptors) {
	printf("-- Saving feature descriptors to [%s] using OpenCV FileStorage\n",
			filename.c_str());
	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		fprintf(stderr, "Could not open file [%s]", filename.c_str());
		return;
	}

	fs << "TotalKeypoints" << descriptors.rows;
	fs << "DescriptorSize" << descriptors.cols; // Recall this is in Bytes
	fs << "DescriptorType" << descriptors.type(); // CV_8U = 0 for binary descriptors

	fs << "KeyPoints" << "{";

	for (int i = 0; i < descriptors.rows; i++) {
		cv::KeyPoint k = keypoints[i];
		fs << "KeyPoint" << "{";
		fs << "x" << k.pt.x;
		fs << "y" << k.pt.y;
		fs << "size" << k.size;
		fs << "angle" << k.angle;
		fs << "response" << k.response;
		fs << "octave" << k.octave;

		fs << "descriptor" << descriptors.row(i);

		fs << "}";
	}

	fs << "}"; // End of structure node

	fs.release();
}

void writeFeaturesToFile(const string& outputFilepath,
		const std::vector<cv::KeyPoint>& keypoints, const Mat& descriptors) {
	printf("-- Saving feature descriptors to [%s] in plain text format\n",
			outputFilepath.c_str());
	std::ofstream outputFile;
	outputFile.open(outputFilepath.c_str(), ios::out | ios::trunc);
	outputFile << descriptors.rows << " " << descriptors.cols << " "
			<< descriptors.type() << std::endl;
	for (int i = 0; i < (int) keypoints.size(); ++i) {
		outputFile << (float) keypoints[i].pt.x << " ";
		outputFile << (float) keypoints[i].pt.y << " ";
		outputFile << (float) keypoints[i].size << " ";
		outputFile << (float) keypoints[i].angle << std::endl;
		for (int j = 0; j < descriptors.cols; ++j) {
			if (descriptors.type() == CV_8U) {
				outputFile << (unsigned int) descriptors.at<uchar>(i, j) << " "; // Print as uint
			} else {
				outputFile << (float) descriptors.at<float>(i, j) << " ";
			}
		}
		outputFile << std::endl;
	}
	outputFile.close();
}

/**
 * Counts the number of bits equal to 1 in a specified number.
 *
 * @param i Number to count bits on
 * @return Number of bits equal to 1
 */
int NumberOfSetBits(int i) {
	i = i - ((i >> 1) & 0x55555555);
	i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
	return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

int BinToDec(const cv::Mat& binRow) {
	if (binRow.type() != CV_8U) {
		fprintf(stderr, "BinToDec: error, received matrix is not binary");
		throw;
	}
	if (binRow.rows != 1) {
		fprintf(stderr,
				"BinToDec: error, received matrix must have only one row");
		throw;
	}
	int decimal = 0;
	for (int i = 0; i < binRow.cols; i++) {
		decimal = decimal * 2 + ((bool) binRow.at<uchar>(0, i));
	}
	return decimal;
}
