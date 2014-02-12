#include <FileUtils.hpp>

#include <algorithm>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>
#include <stdlib.h>

#include <sys/stat.h>

void FileUtils::readFolder(const char* folderPath,
		std::vector<std::string>& files) {
	DIR *dir;
	struct dirent *ent;
	// Try opening folder
	if ((dir = opendir(folderPath)) != NULL) {
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
	} else {
		throw std::runtime_error(
				"Could not open directory [" + std::string(folderPath) + "]");
	}
}

// --------------------------------------------------------------------------

void FileUtils::saveList(const std::string& list_fpath,
		const std::vector<std::string>& list) {

	// Open file
	std::ofstream outputFileStream(list_fpath.c_str(), std::fstream::out);

	// Check file
	if (outputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + list_fpath + "] for writing\n");
	}

	// Save list to file
	for (const std::string& line : list) {
		outputFileStream << line << std::endl;
	}

	// Close file
	outputFileStream.close();
}

// --------------------------------------------------------------------------

void FileUtils::loadList(const std::string& list_fpath,
		std::vector<std::string>& list) {

	// Initializing variables
	std::ifstream inputFileStream;
	std::string line;
	list.clear();

	// Open file
	inputFileStream.open(list_fpath.c_str(), std::fstream::in);

	// Check file
	if (inputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + list_fpath + "] for reading");
	}

	// Load list from file
	while (getline(inputFileStream, line)) {
		list.push_back(line);
	}

	// Close file
	inputFileStream.close();
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

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveKeypoints] "
				"Unable to open file [" + filename + "] for writing");
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

	fs << "]";

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
		throw std::runtime_error("[FileUtils::loadFeatures] "
				"Unable to open file [" + filename + "] for reading");
	}

	int rows, cols, type;

	rows = (int) fs["TotalKeypoints"];
	cols = (int) fs["DescriptorSize"];
	type = (int) fs["DescriptorType"];

	descriptors.create(rows, cols, type);
	keypoints.reserve(rows);

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[FileUtils::loadFeatures] "
				"Fetched element 'KeyPoints' is not a sequence");
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

void FileUtils::saveDescriptors(const std::string& filename,
		const cv::Mat& descriptors) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveDescriptors] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "Descriptors" << descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptors(const std::string& filename,
		cv::Mat& descriptors) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::loadDescriptors] "
				"Unable to open file [" + filename + "] for reading");
	}

	descriptors.release();
	descriptors = cv::Mat();

	fs["Descriptors"] >> descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::saveKeypoints(const std::string& filename,
		const std::vector<cv::KeyPoint>& keypoints) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::saveKeypoints] "
				"Unable to open file [" + filename + "] for writing");
	}

	fs << "KeyPoints" << "[";

	for (size_t i = 0; i < keypoints.size(); i++) {
		cv::KeyPoint k = keypoints[i];
		fs << "{:";
		fs << "x" << k.pt.x;
		fs << "y" << k.pt.y;
		fs << "size" << k.size;
		fs << "angle" << k.angle;
		fs << "response" << k.response;
		fs << "octave" << k.octave;
		fs << "}";
	}

	fs << "]";

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::loadKeypoints(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[FileUtils::loadKeypoints] "
				"Unable to open file [" + filename + "] for reading");
	}

	cv::FileNode keypointsSequence = fs["KeyPoints"];

	if (keypointsSequence.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[FileUtils::loadKeypoints] "
				"Fetched element 'KeyPoints' is not a sequence");
	}

	keypoints.clear();

	for (cv::FileNodeIterator it = keypointsSequence.begin();
			it != keypointsSequence.end(); it++) {
		keypoints.push_back(
				cv::KeyPoint((float) (*it)["x"], (float) (*it)["y"],
						(float) (*it)["size"], (float) (*it)["angle"],
						(float) (*it)["response"], (int) (*it)["octave"]));
	}

	fs.release();

}

// --------------------------------------------------------------------------

bool FileUtils::checkFileExist(const std::string& fname) {
	struct stat buffer;
	return stat(fname.c_str(), &buffer) == 0;
}

// --------------------------------------------------------------------------

void FileUtils::loadQueriesList(std::string& filePath,
		std::vector<Query>& list) {

	// Initialize local variables
	list.clear();
	std::ifstream inputFileStream;
	Query query;

	// Open file
	inputFileStream.open(filePath.c_str(), std::fstream::in);

	// Check file
	if (inputFileStream.good() == false) {
		throw std::runtime_error(
				"Error while opening file [" + filePath + "] for reading");
	}

	// Load list from file
	while (inputFileStream >> query.name >> query.x1 >> query.y1 >> query.x2
			>> query.y2) {
		list.push_back(query);
		// Clear variable holding temporary query
		query.clear();
	}

	// Close file
	inputFileStream.close();

}
