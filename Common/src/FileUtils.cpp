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
		throw std::runtime_error("[FileUtils::loadKeypoints] "
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
		throw std::runtime_error("[FileUtils::loadKeypoints] "
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
		throw std::runtime_error("[FileUtils::saveKeypoints] "
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
		throw std::runtime_error("[FileUtils::loadKeypoints] "
				"Unable to open file [" + filename + "] for reading");
	}

	descriptors = cv::Mat();

	fs["Descriptors"] >> descriptors;

	fs.release();

}

// --------------------------------------------------------------------------

void FileUtils::saveDescriptorsToBin(const std::string& filename,
		const cv::Mat& descriptors) {

	cv::Ptr<FILE> filePtr = fopen(filename.c_str(), "wb");

	if (filePtr.empty() == true) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

	fwrite(descriptors.data, descriptors.rows * descriptors.cols,
			descriptors.elemSize(), filePtr);

}

// --------------------------------------------------------------------------

void FileUtils::loadDescriptorsFromBin(const std::string& filename,
		cv::Mat& descriptors, int descriptorLength) {

	FILE * filePtr;
	long fileSize;
	char * buffer;
	size_t result;

	filePtr = fopen(filename.c_str(), "rb");
	if (filePtr == NULL) {
		throw std::runtime_error("Could not open file [" + filename + "]");
	}

	// Obtain file size
	fseek(filePtr, 0, SEEK_END);
	fileSize = ftell(filePtr);
	rewind(filePtr);

	// Allocate memory to contain the whole file
	buffer = (char*) malloc(sizeof(char) * fileSize);

	if (buffer == NULL) {
		throw std::runtime_error("Memory error");
	}

	// Copy the file into the buffer
	result = fread(buffer, 1, fileSize, filePtr);

	if (long(result) != fileSize) {
		throw std::runtime_error("Reading error");
	}

	// clean up
	fclose(filePtr);
	free(buffer);

	descriptors = cv::Mat();

//	Mat image;
//	uint16_t *imageMap = (uint16_t*) buffer;
//	image.create(rows, cols, CV_16UC1);
//	memcpy(image.data, imageMap, rows * cols * sizeof(uint16_t));

//	float* data = new float[100 * 100];
//	Mat src(100, 100, CV_32FC1, data);
//	delete [] data;
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
