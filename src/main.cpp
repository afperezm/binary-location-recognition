//============================================================================
// Name        : MediaEval-PlacingTask.cpp
// Author      : Andrés Pérez
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <ctime>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/internal.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <AgastFeatureDetector.h>
#include <DBriefDescriptorExtractor.h>
#include <KMajority.h>

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
 *
 * @param filename
 * @param keypoints vector of keypoints
 * @param descriptors matrix of descriptors
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

namespace cv {

CV_INIT_ALGORITHM(AgastFeatureDetector, "Feature2D.AGAST",
		obj.info()->addParam(obj, "threshold", obj.threshold); obj.info()->addParam(obj, "nonmaxsuppression", obj.nonmaxsuppression); obj.info()->addParam(obj, "type", obj.type))
;

CV_INIT_ALGORITHM(DBriefDescriptorExtractor, "Feature2D.DBRIEF", obj.info())
;

}

int main(int argc, char **argv) {

	if (argc != 2) {
		printf("\n");
		printf("Usage: %s <img1>", argv[0]);
		printf("\n\n");
		return EXIT_FAILURE;
	}

	// Initiating nonfree module, it's necessary for using SIFT and SURF
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

	Mat descriptors_1;

	mytime = cv::getTickCount();
	extractor->compute(img_1, keypoints_1, descriptors_1);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	// Notice that number of keypoints might be reduced due to border effect
	printf(
			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
			descriptors_1.rows, descriptors_1.cols,
			descriptors_1.type() == CV_8U ? "binary" : "real-valued", mytime);

	// Step 3/5: show keypoints

	cv::drawKeypoints(img_1, keypoints_1, img_1, cv::Scalar::all(-1));
	cv::namedWindow("Image keypoints", CV_WINDOW_NORMAL);
	cv::imshow("Image keypoints", img_1);

	cv::waitKey(0);

	// Step 4/5: save descriptors into a file for later use
	save("test.xml.gz", keypoints_1, descriptors_1);
	writeFeaturesToFile("test_descriptors", keypoints_1, descriptors_1);

	// Step 5/5: cluster descriptors

	std::srand(unsigned(std::time(0)));
	cv::Ptr<KMajority> obj = new KMajority(16, 100);
	mytime = cv::getTickCount();
	obj->cluster(keypoints_1, descriptors_1);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("-- Clustered [%zu] keypoints in [%d] clusters in [%lf] ms\n",
			keypoints_1.size(), obj->getNumberOfClusters(), mytime);

	for (uint j = 0; j < obj->getNumberOfClusters(); j++) {
//		cout << obj->getCentroids().row(j) << endl;
//		printDescriptors(obj->getCentroids().row(j));
		printf("   Cluster %u has %u transactions assigned\n", j + 1,
				obj->getClusterCounts()[j]);
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
		fprintf(stderr,
				(string("Could not open file [") + filename + string("]")).c_str());
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
