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

void detectAndDescribeFeatures(const std::string& imgPath,
		const std::string& imgName, std::vector<cv::KeyPoint>& keypoints,
		cv::Mat& descriptors, const std::string detectorType = "SIFT",
		const std::string descriptorType = "BRIEF");

double mytime;

int main(int argc, char **argv) {

	if (argc < 3 || argc > 5) {
		printf(
				"\nUsage:\n"
						"\t%s <in.imgs.folder> <out.keys.folder> [in.detector:SIFT] [in.descriptor:BRISK]\n\n",
				argv[0]);
		return EXIT_FAILURE;
	}

	char* imgsFolder = argv[1];
	char* keysFolder = argv[2];
	std::string detectorType = "SIFT";
	std::string descriptorType = "BRISK";

	if (argc >= 4) {
		detectorType = std::string(argv[3]);
	}

	if (argc >= 5) {
		descriptorType = std::string(argv[4]);
	}

	cv::initModule_features2d_extensions();

	// Initiating non-free module, it's necessary for using SIFT and SURF
	cv::initModule_nonfree();

	// Verifying that input detector/descriptor are valid OpenCV algorithms
	std::vector<std::string> algorithms;
	cv::Algorithm::getList(algorithms);
	bool isDetectorValid = false, isDescriptorValid = false;

	for (std::string algName : algorithms) {
		isDetectorValid |= algName.substr(
				algName.size() < detectorType.size() ?
						0 : algName.size() - detectorType.size()).compare(
				detectorType) == 0;
		isDescriptorValid |= algName.substr(
				algName.size() < descriptorType.size() ?
						0 : algName.size() - descriptorType.size()).compare(
				descriptorType) == 0;
	}
	if (isDetectorValid == false) {
		fprintf(stderr,
				"Input detector=[%s] is not a registered algorithm in OpenCV\n",
				detectorType.c_str());
		return EXIT_FAILURE;
	}
	if (isDescriptorValid == false) {
		fprintf(stderr,
				"Input descriptor=[%s] is not a registered algorithm in OpenCV\n",
				descriptorType.c_str());
		return EXIT_FAILURE;
	}

	printf("-- Using detector=[%s] descriptor=[%s]\n", detectorType.c_str(),
			descriptorType.c_str());

	printf("-- Loading images in folder [%s]\n", imgsFolder);

	// Load files from images folder into a vector
	std::vector<std::string> imgFolderFiles;
	try {
		FileUtils::readFolder(imgsFolder, imgFolderFiles);
	} catch (const std::runtime_error& error) {
		fprintf(stderr, "%s\n", error.what());
		return EXIT_FAILURE;
	}
	std::sort(imgFolderFiles.begin(), imgFolderFiles.end());

	// Finding last written key file
	printf(
			"-- Searching for previous written key files in [%s] to resume feature extraction\n",
			keysFolder);
	// By default set to the first file
	std::vector<std::string>::iterator startImg = imgFolderFiles.begin();
	// Load files from keys folder into a vector
	std::vector<std::string> keyFiles;
	try {
		FileUtils::readFolder(keysFolder, keyFiles);
	} catch (const std::runtime_error& error) {
		fprintf(stderr, "%s\n", error.what());
		return EXIT_FAILURE;
	}

	// Sort loaded files in reverse order
	std::sort(keyFiles.begin(), keyFiles.end(), std::greater<std::string>());
	// Search for a valid key file
	for (std::string& keyFile : keyFiles) {
		// Check if found a key file, i.e. a file with yaml.gz extension
		if (keyFile.find("yaml.gz") != std::string::npos) {
			std::vector<std::string>::iterator result = std::find(
					imgFolderFiles.begin(), imgFolderFiles.end(),
					keyFile.substr(0, keyFile.size() - 8) + ".jpg");
			if (result != imgFolderFiles.end()) {
				// Found element
				startImg = result;
				printf("   Restarting feature extraction from image [%s]\n",
						(*startImg).c_str());
				break;
			}
		}
	}

	if (startImg == imgFolderFiles.begin()) {
		printf("   Starting feature extraction from first image\n");
	}

	for (std::vector<std::string>::iterator image = startImg;
			image != imgFolderFiles.end(); ++image) {
		if ((*image).find(".jpg") != std::string::npos) {
			printf("-- Processing image [%s]\n", (*image).c_str());

			std::vector<cv::KeyPoint> keypoints;
			cv::Mat descriptors;
			try {
				// Notice that number of keypoints might be reduced due to border effect
				detectAndDescribeFeatures(imgsFolder, (*image), keypoints,
						descriptors, detectorType, descriptorType);
				CV_Assert((int )keypoints.size() == descriptors.rows);
			} catch (const std::runtime_error& error) {
				fprintf(stderr, "%s\n", error.what());
				return EXIT_FAILURE;
			}

			// Show keypoints

			//	cv::drawKeypoints(img_1, keypoints, img_1, cv::Scalar::all(-1));
			//	cv::namedWindow("Image keypoints", CV_WINDOW_NORMAL);
			//	cv::imshow("Image keypoints", img_1);
			//	cv::waitKey(0);

			std::string descriptorFileName(keysFolder);
			descriptorFileName += "/" + (*image).substr(0, (*image).size() - 4)
					+ ".yaml.gz";

			printf(
					"-- Saving feature descriptors to [%s] using OpenCV FileStorage\n",
					descriptorFileName.c_str());

			try {
				FileUtils::saveFeatures(descriptorFileName, keypoints,
						descriptors);
			} catch (const std::runtime_error& error) {
				fprintf(stderr, "%s\n", error.what());
				return EXIT_FAILURE;
			}

			descriptors.release();
		}
	}

	return EXIT_SUCCESS;
}

void detectAndDescribeFeatures(const std::string& imgPath,
		const std::string& imgName, std::vector<cv::KeyPoint>& keypoints,
		cv::Mat& descriptors, const std::string detectorType,
		const std::string descriptorType) {

	cv::Mat img = cv::imread(imgPath + std::string("/") + imgName,
			CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		throw std::runtime_error(
				"Error reading image [" + imgPath + std::string("/") + imgName
						+ "]");
	} else {
		// Create smart pointer for feature detector
		cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
				detectorType);

		keypoints.clear();
		// Detect the keypoints
		printf("   Detecting keypoints from image [%s]\n", imgName.c_str());
		detector->detect(img, keypoints);

		//	mytime = cv::getTickCount();
		//	detector->detect(img_1, keypoints);
		//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
		//			* 1000;
		//	printf("-- Detected [%zu] keypoints in [%lf] ms\n", keypoints.size(),
		//			mytime);

		// Create smart pointer for descriptor extractor
		cv::Ptr<cv::DescriptorExtractor> extractor =
				cv::DescriptorExtractor::create(descriptorType);

		descriptors = cv::Mat();
		// Describe keypoints
		// Notice that number of keypoints might be reduced due to border effect
		printf("   Describing keypoints from image [%s]\n", imgName.c_str());
		extractor->compute(img, keypoints, descriptors);

		img.release();

		//	mytime = cv::getTickCount();
		//	extractor->compute(img_1, keypoints, descriptors);
		//	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
		//			* 1000;

		// Notice that number of keypoints might be reduced due to border effect
		//	printf(
		//			"-- Extracted [%d] descriptors of size [%d] and type [%s] in [%lf] ms\n",
		//			descriptors.rows, descriptors.cols,
		//			descriptors.type() == CV_8U ? "binary" : "real-valued", mytime);
	}

}
