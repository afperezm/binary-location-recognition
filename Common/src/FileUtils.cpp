#include <FileUtils.hpp>

#include <algorithm>
#include <dirent.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <sys/stat.h>

void FileUtils::readFolder(const char* folderPath,
		std::vector<std::string>& files) {
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
		std::sort(files.begin(), files.end());
		fprintf(stdout, "   Found [%d] files\n", (int) files.size());
	} else {
		throw std::runtime_error(
				"Could not open directory [" + std::string(folderPath) + "]");
	}
}

// --------------------------------------------------------------------------

void FileUtils::saveFeatures(const std::string &filename,
		const std::vector<cv::KeyPoint>& keypoints,
		const cv::Mat& descriptors) {

#if FILEUTILSVERBOSE
	printf(
			"-- Saving feature descriptors to [%s] using OpenCV FileStorage\n",
			filename.c_str());
#endif

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (!fs.isOpened()) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

	fs << "TotalKeypoints" << descriptors.rows;
	fs << "DescriptorSize" << descriptors.cols; // In Bytes for binary descriptors
	fs << "DescriptorType" << descriptors.type(); // CV_8U = 0 for binary descriptors

	fs << "KeyPoints" << "[";

	for (int i = 0; i < descriptors.rows; i++) {
		cv::KeyPoint k = keypoints[i];
		fs << "{";
		fs << "x" << k.pt.x;
		fs << "y" << k.pt.y;
		fs << "size" << k.size;
		fs << "angle" << k.angle;
		fs << "response" << k.response;
		fs << "octave" << k.octave;

		fs << "descriptor" << descriptors.row(i);

		fs << "}";
	}

	fs << "]"; // End of structure node

	fs.release();
}

// --------------------------------------------------------------------------

void FileUtils::loadFeatures(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors) {

#if FILEUTILSVERBOSE
	printf(
			"-- Loading feature descriptors from [%s] using OpenCV FileStorage\n",
			filename.c_str());
#endif

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

	int rows, cols, type;

	rows = (int) fs["TotalKeypoints"];
	cols = (int) fs["DescriptorSize"];
	type = (int) fs["DescriptorType"];

	descriptors.create(rows, cols, type);
	keypoints.reserve(rows);

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("Error while parsing [" + filename + "]"
				" fetched element KeyPoints is not a sequence");
	}

	int idx = 0;

	cv::Mat featureVector;

	for (cv::FileNodeIterator it = keypointsSequence.begin();
			it != keypointsSequence.end(); it++, idx++) {

		keypoints.push_back(
				cv::KeyPoint((float) (*it)["x"], (float) (*it)["y"],
						(float) (*it)["size"], (float) (*it)["angle"],
						(float) (*it)["response"], (int) (*it)["octave"]));

		featureVector = descriptors.row(idx);

		(*it)["descriptor"] >> featureVector;

	}

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::saveDescriptors(const std::string& filename, const cv::Mat& descriptors){

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (!fs.isOpened()) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

/*
	fs << "Total" << descriptors.rows;
	fs << "Size" << descriptors.cols; // In Bytes for binary descriptors
	fs << "Type" << descriptors.type(); // CV_8U = 0 for binary descriptors
*/

	fs << "Descriptors" << descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptors(const std::string& filename, cv::Mat& descriptors){

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

/*
	int rows, cols, type;

	rows = (int) fs["Total"];
	cols = (int) fs["Size"];
	type = (int) fs["Type"];

	descriptors.create(rows, cols, type);
*/

	fs["Descriptors"] >> descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

bool FileUtils::checkFileExist(const std::string& fname) {
	struct stat buffer;
	return stat(fname.c_str(), &buffer) == 0;
}
