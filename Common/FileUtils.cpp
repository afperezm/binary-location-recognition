#include <FileUtils.hpp>

#include <algorithm>
#include <dirent.h>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>

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
		std::stringstream ss;
		ss << "Could not open directory [" << folderPath << "]";
		throw std::runtime_error(ss.str());
	}
}

// --------------------------------------------------------------------------

void FileUtils::saveFeatures(const std::string &filename,
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

	printf(
			"-- Loading feature descriptors from [%s] using OpenCV FileStorage\n",
			filename.c_str());

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		std::stringstream ss;
		ss << "Could not open file [" << filename << "]";
		throw std::runtime_error(ss.str());
	}

	int rows, cols, type;

	rows = (int) fs["TotalKeypoints"];
	cols = (int) fs["DescriptorSize"];
	type = (int) fs["DescriptorType"];

	descriptors.create(rows, cols, type);
	keypoints.reserve(rows);

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		std::stringstream ss;
		ss << "Error while parsing [" << filename;
		ss << "] fetched element KeyPoints is not a sequence";
		throw std::runtime_error(ss.str());
	}

	int idx = 0;

	for (cv::FileNodeIterator it = keypointsSequence.begin();
			it != keypointsSequence.end(); it++, idx++) {

		keypoints.push_back(
				cv::KeyPoint((float) (*it)["x"], (float) (*it)["y"],
						(float) (*it)["size"], (float) (*it)["angle"],
						(float) (*it)["response"], (int) (*it)["octave"]));

		cv::Mat descriptor;
		(*it)["descriptor"] >> descriptor;

		descriptor.copyTo(descriptors.row(idx));

	}

	fs.release();

}
