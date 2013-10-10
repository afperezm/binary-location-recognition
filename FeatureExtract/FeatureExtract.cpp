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

#include <opencv2/core/internal.hpp>
#include <opencv2/extensions/features2d.hpp>
#include <opencv2/flann/logger.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

void detectAndDescribeFeatures(const std::string& imgPath,
		const std::string& imgName, std::vector<cv::KeyPoint>& keypoints,
		cv::Mat& descriptors);

/**
 * Opens a directory and saves all the files names onto a vector of strings, it
 * returns a status flag for reporting any error during the opening of the folder.
 *
 * @param folderPath - Path to the folder to be opened
 * @param files - Reference to a vector where all the files names will be saved
 */
void readFolder(const char* folderfolderPath, std::vector<std::string>& files);

/**
 * Saves a set of features (keypoints and descriptors) onto a plain text
 * file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to save the features
 * @param keypoints - The keypoints to be saved
 * @param descriptors - The descriptors to be saved
 */
void save(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors);

double mytime;

int main(int argc, char **argv) {

	cv::initModule_features2d_extensions();

	if (argc != 3) {
		printf("\nUsage:\n"
				"\t%s <in.imgs.folder> <out.keys.folder>\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	char* imgsFolder = argv[1];
	char* keysFolder = argv[2];

	// Initiating non-free module, it's necessary for using SIFT and SURF
	cv::initModule_nonfree();

	printf("-- Loading images in folder [%s]\n", imgsFolder);

	std::vector<std::string> imgFolderFiles;
	try {
		readFolder(imgsFolder, imgFolderFiles);
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
	// Load files in keys folder
	std::vector<std::string> keyFiles;
	try {
		readFolder(keysFolder, keyFiles);
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
				detectAndDescribeFeatures(imgsFolder, (*image), keypoints,
						descriptors);
				CV_Assert((int )keypoints.size() == descriptors.rows);
			} catch (const std::runtime_error& error) {
				fprintf(stderr, "%s\n", error.what());
				return EXIT_FAILURE;
			}

			std::string descriptorFileName(keysFolder);
			descriptorFileName += "/" + (*image).substr(0, (*image).size() - 4)
					+ ".yaml.gz";

			printf("   Saving feature descriptors to [%s]\n",
					descriptorFileName.c_str());

			try {
				save(descriptorFileName, keypoints, descriptors);
			} catch (const std::runtime_error& error) {
				fprintf(stderr, "%s\n", error.what());
				return EXIT_FAILURE;
			}
		}
	}

	return EXIT_SUCCESS;
}

void detectAndDescribeFeatures(const std::string& imgPath,
		const std::string& imgName, std::vector<cv::KeyPoint>& keypoints,
		cv::Mat& descriptors) {

	cv::Mat img = cv::imread(imgPath + std::string("/") + imgName, CV_LOAD_IMAGE_GRAYSCALE);

	if (!img.data) {
		std::stringstream ss;
		ss << "Error reading image [" << imgPath + std::string("/") + imgName << "]";
		throw std::runtime_error(ss.str());
	} else {
		// Create smart pointer for feature detector
		cv::Ptr<cv::FeatureDetector> detector = cv::FeatureDetector::create(
				"AGAST");

		keypoints.clear();
		// Detect the keypoints
		printf("   Detecting keypoints from image [%s]\n", imgName.c_str());
		detector->detect(img, keypoints);

		// Create smart pointer for descriptor extractor
		cv::Ptr<cv::DescriptorExtractor> extractor =
				cv::DescriptorExtractor::create("BRIEF");

		descriptors = cv::Mat();
		// Describe keypoints
		printf("   Describing keypoints from image [%s]\n", imgName.c_str());
		extractor->compute(img, keypoints, descriptors);
	}

}

void readFolder(const char* folderPath, std::vector<std::string>& files) {
	DIR *dir;
	struct dirent *ent;
	// Try opening folder
	if ((dir = opendir(folderPath)) != NULL) {
		fprintf(stdout, "   Opening directory [%s]\n", folderPath);
		// Save all true directory names into a vector of strings
		while ((ent = readdir(dir)) != NULL) {
			// Ignore . and .. as valid folder names
			std::string name = std::string(ent->d_name);
			if (name.compare(".") != 0 && name.compare("..") != 0) {
				files.push_back(std::string(ent->d_name));
			}
		}
		closedir(dir);
		// Sort alphabetically vector of folder names
		sort(files.begin(), files.end());
		fprintf(stdout, "   Found [%d] files\n", (int) files.size());
	} else {
		std::stringstream ss;
		ss << "Could not open directory [" << folderPath << "]";
		throw std::runtime_error(ss.str());
	}
}

void save(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);
	if (!fs.isOpened()) {
		std::stringstream ss;
		ss << "Could not open file [" << filename << "]";
		throw std::runtime_error(ss.str());
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
