/*
 * FeatExtract.cpp
 *
 *  Created on: Oct 8, 2013
 *      Author: andresf
 */

#include <dirent.h>
#include <fstream>
#include <stdexcept>
#include <stdlib.h>
#include <string>
#include <vector>

#include <FileUtils.hpp>

#include <opencv2/core/internal.hpp>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/flann/logger.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

double mytime;

/**
 * Verify candidate algorithm is a valid OpenCV algorithm.
 *
 * @param candidateAlgorithm - Name of the candidate algorithm
 * @return true if valid, false otherwise
 */
bool isValidAlgorithm(std::string& candidateAlgorithm);

/**
 * Detect features from an image using some OpenCV supported detector.
 *
 * @param imgPath - Path to the images folder
 * @param imgName - Name of the image
 * @param keyPoints - Vector of key-points
 * @param detectorType - Detector algorithm name
 */
void detectFeatures(const std::string& imgPath, const std::string& imgName,
		std::vector<cv::KeyPoint>& keyPoints, const std::string detectorType);

/**
 * Extract descriptor from the at the indicated key-point positions.
 *
 * @param imgPath - Path to the images folder
 * @param imgName - Name of the image
 * @param keyPoints - Vector of key-points
 * @param descriptors - Matrix where to save extracted descriptors
 * @param descriptorType - Descriptor algorithm name
 */
void describeFeatures(const std::string& imgPath, const std::string& imgName,
		std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors,
		const std::string descriptorType);

/**
 * In a given folder finds the last written .yaml.gz file that is also a valid image file.
 *
 * @param folderPath - Path to the folder where to search
 * @param imgFolderFiles - Vector of images names
 *
 * @return an iterator to the image where the last written file is positioned
 */
std::vector<std::string>::iterator findLastWrittenFile(
		const std::string& folderPath,
		std::vector<std::string>& imgFolderFiles);

int main(int argc, char **argv) {

	if (argc < 5 || argc > 6) {
		printf(
				"\nUsage:\n"
						"\tFeatureExtract -detect <in.detector.type> <in.imgs.folder> <out.keypoints.folder>\n"
						"\tFeatureExtract -extract <in.descriptor.type> <in.imgs.folder> <in.keypoints.folder> <out.descriptors.folder>\n\n");
		return EXIT_FAILURE;
	}

	// Initialize 2d features extensions mode, necessary for using AGAST and DBRIEF
	cv::initModule_features2d_extensions();

	// Initialize non-free module, necessary for using SIFT and SURF
	cv::initModule_nonfree();

	std::string option = argv[1];

	if (option.compare("-detect") == 0) {
		std::string detectorType = argv[2];
		std::string imgsFolder = argv[3];
		std::string keypointsFolder = argv[4];

		bool isDetectorValid = isValidAlgorithm(detectorType);

		if (isDetectorValid == false) {
			fprintf(stderr,
					"Input detector=[%s] is not a registered algorithm in OpenCV\n",
					detectorType.c_str());
			return EXIT_FAILURE;
		}

		printf("-- Using [%s] detector\n", detectorType.c_str());

		/* Load files from images folder into a vector */
		printf("-- Loading images in folder [%s]\n", imgsFolder.c_str());

		std::vector<std::string> imgFolderFiles;
		try {
			printf("   Opening directory [%s]\n", imgsFolder.c_str());
			FileUtils::readFolder(imgsFolder.c_str(), imgFolderFiles);
			printf("   Found [%lu] files\n", imgFolderFiles.size());
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

		/* Finding last written key-points file */
		printf(
				"-- Searching for previous written key files in [%s] to resume feature extraction\n",
				keypointsFolder.c_str());

		// By default set to the first file
		std::vector<std::string>::iterator startImg = findLastWrittenFile(
				keypointsFolder, imgFolderFiles);

		if (startImg == imgFolderFiles.begin()) {
			printf("   Starting feature extraction from first image\n");
		} else {
			printf("   Restarting feature extraction from image [%s]\n",
					(*startImg).c_str());
		}

		std::vector<cv::KeyPoint> keypoints;

		/* Extracting features */
		for (std::vector<std::string>::iterator image = startImg;
				image != imgFolderFiles.end(); ++image) {
			if ((*image).find(".jpg") != std::string::npos) {
				printf("-- Processing image [%s]\n", (*image).c_str());

				try {
					// Note: number of key-points might be reduced due to border effect
					detectFeatures(imgsFolder, *image, keypoints, detectorType);
				} catch (const std::runtime_error& error) {
					fprintf(stderr, "%s\n", error.what());
					return EXIT_FAILURE;
				}

				std::string keypointsFileName = keypointsFolder + "/"
						+ (*image).substr(0, (*image).size() - 4) + ".yaml.gz";

				printf(
						"-- Saving feature key-points to [%s] using OpenCV FileStorage\n",
						keypointsFileName.c_str());

				try {
					FileUtils::saveKeypoints(keypointsFileName, keypoints);
				} catch (const std::runtime_error& error) {
					fprintf(stderr, "%s\n", error.what());
					return EXIT_FAILURE;
				}

			}
		}

	} else if (option.compare("-extract") == 0) {
		std::string descriptorType = argv[2];
		std::string imgsFolder = argv[3];
		std::string keypointsFolder = argv[4];
		std::string descriptorsFolder = argv[5];

		bool isDescriptorValid = isValidAlgorithm(descriptorType);

		if (isDescriptorValid == false) {
			fprintf(stderr,
					"Input descriptor=[%s] is not a registered algorithm in OpenCV\n",
					descriptorType.c_str());
			return EXIT_FAILURE;
		}

		printf("-- Using [%s] descriptor\n", descriptorType.c_str());

		/* Load files from images folder into a vector */
		printf("-- Loading images in folder [%s]\n", imgsFolder.c_str());

		std::vector<std::string> imgFolderFiles;
		try {
			printf("   Opening directory [%s]\n", imgsFolder.c_str());
			FileUtils::readFolder(imgsFolder.c_str(), imgFolderFiles);
			printf("   Found [%lu] files\n", imgFolderFiles.size());
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

		/* Finding last written descriptors file */
		printf(
				"-- Searching for previous written key files in [%s] to resume feature extraction\n",
				keypointsFolder.c_str());

		// By default set to the first file
		std::vector<std::string>::iterator startImg = findLastWrittenFile(
				descriptorsFolder, imgFolderFiles);

		if (startImg == imgFolderFiles.begin()) {
			printf("   Starting feature extraction from first image\n");
		} else {
			printf("   Restarting feature extraction from image [%s]\n",
					(*startImg).c_str());
		}

		std::vector<cv::KeyPoint> keypoints;
		cv::Mat descriptors;

		/* Extracting features */
		for (std::vector<std::string>::iterator image = startImg;
				image != imgFolderFiles.end(); ++image) {
			if ((*image).find(".jpg") != std::string::npos) {
				printf("-- Processing image [%s]\n", (*image).c_str());

				std::string keypointsFileName = keypointsFolder + "/"
						+ (*image).substr(0, (*image).size() - 4) + ".yaml.gz";

				printf(
						"   Loading feature key-points from [%s] using OpenCV FileStorage\n",
						keypointsFileName.c_str());

				try {
					FileUtils::loadKeypoints(keypointsFileName, keypoints);
				} catch (const std::runtime_error& error) {
					fprintf(stderr, "%s\n", error.what());
					return EXIT_FAILURE;
				}

				try {
					// Note: number of key-points might be reduced due to border effect
					describeFeatures(imgsFolder, *image, keypoints, descriptors,
							descriptorType);
					CV_Assert(
							descriptors.rows >= 0
									&& keypoints.size()
											== (size_t )descriptors.rows);
				} catch (const std::runtime_error& error) {
					fprintf(stderr, "%s\n", error.what());
					return EXIT_FAILURE;
				}

				std::string descriptorFileName = descriptorsFolder + "/"
						+ (*image).substr(0, (*image).size() - 4) + ".yaml.gz";

				printf(
						"   Saving feature descriptors to [%s] using OpenCV FileStorage\n",
						descriptorFileName.c_str());

				try {
					FileUtils::saveDescriptors(descriptorFileName, descriptors);
				} catch (const std::runtime_error& error) {
					fprintf(stderr, "%s\n", error.what());
					return EXIT_FAILURE;
				}

			}
		}

	} else {
		fprintf(stderr, "Invalid option chosen\n");
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}

bool isValidAlgorithm(std::string& candidateAlgorithm) {

	std::vector<std::string> algorithms;
	cv::Algorithm::getList(algorithms);

	bool isValid = false;

	for (std::string& algorithm : algorithms) {
		isValid |=
				algorithm.substr(
						algorithm.size() < candidateAlgorithm.size() ?
								0 :
								algorithm.size() - candidateAlgorithm.size()).compare(
						candidateAlgorithm) == 0;
	}

	return isValid;
}

void detectFeatures(const std::string& imgPath, const std::string& imgName,
		std::vector<cv::KeyPoint>& keyPoints, const std::string detectorType) {

	cv::Mat img = cv::imread(imgPath + std::string("/") + imgName,
			CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		throw std::runtime_error(
				"Error while reading image [" + imgPath + "/" + imgName + "]");
	}

	// Create smart pointer for feature detector
	cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
			detectorType);

	// Clear key-points
	std::vector<cv::KeyPoint>().swap(keyPoints);
	// Detect the key-points
	printf("   Detecting key-points from image [%s]\n", imgName.c_str());
	detector->detect(img, keyPoints);

	//	mytime = cv::getTickCount();
	//	detector->detect(img_1, key-points);
	//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
	//			* 1000;
	//	printf("-- Detected [%zu] key-points in [%lf] ms\n", keypoints.size(),
	//			mytime);

	img.release();
	img = cv::Mat();

}

void describeFeatures(const std::string& imgPath, const std::string& imgName,
		std::vector<cv::KeyPoint>& keyPoints, cv::Mat& descriptors,
		const std::string descriptorType) {

	cv::Mat img = cv::imread(imgPath + std::string("/") + imgName,
			CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		throw std::runtime_error(
				"Error reading image [" + imgPath + std::string("/") + imgName
						+ "]");
	}

	// Create smart pointer for descriptor extractor
	cv::Ptr<cv::DescriptorExtractor> extractor =
			cv::DescriptorExtractor::create(descriptorType);

	descriptors.release();
	descriptors = cv::Mat();

	// Describe key-points
	// Notice that number of key-points might be reduced due to border effect
	printf("   Describing key-points from image [%s]\n", imgName.c_str());
	extractor->compute(img, keyPoints, descriptors);

	//	mytime = cv::getTickCount();
	//	extractor->compute(img_1, key-points, descriptors);
	//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
	//			* 1000;

	// Note: number of key-points might be reduced due to border effect
	//	printf(
	//			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
	//			descriptors.rows, descriptors.cols,
	//			descriptors.type() == CV_8U ? "binary" : "real-valued", mytime);

	img.release();
	img = cv::Mat();

}

std::vector<std::string>::iterator findLastWrittenFile(
		const std::string& folderPath,
		std::vector<std::string>& imgFolderFiles) {

	// By default set to the first file
	std::vector<std::string>::iterator lastWrittenFile = imgFolderFiles.begin();

	// Load files from folder into a vector
	std::vector<std::string> folderFiles;
	try {
		FileUtils::readFolder(folderPath.c_str(), folderFiles);
	} catch (const std::runtime_error& error) {
		throw std::runtime_error(error.what());
	}

	// Sort loaded files in reverse order
	std::sort(folderFiles.begin(), folderFiles.end(),
			std::greater<std::string>());

	// Search for a file with yaml.gz extension
	for (std::string& folderFile : folderFiles) {
		if (folderFile.find("yaml.gz") != std::string::npos) {
			std::vector<std::string>::iterator result = std::find(
					imgFolderFiles.begin(), imgFolderFiles.end(),
					folderFile.substr(0, folderFile.size() - 8) + ".jpg");
			if (result != imgFolderFiles.end()) {
				// Found element
				lastWrittenFile = result;
				break;
			}
		}
	}

	return lastWrittenFile;
}
