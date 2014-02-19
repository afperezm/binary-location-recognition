#ifndef __file_utils_h__
#define __file_utils_h__

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace FileUtils {

struct Query {

	std::string name;
	float x1, y1, x2, y2;

	Query() {
		clear();
	}

	Query(std::string& _name, float _x1, float _y1, float _x2, float _y2) {
		name = _name;
		x1 = _x1;
		y1 = _y1;
		y2 = _y2;
		x2 = _x2;
	}

	void clear() {
		name.clear();
		x1 = -1.0;
		y1 = -1.0;
		y2 = -1.0;
		x2 = -1.0;
	}

};

struct MatStats {

	int rows;
	int cols;
	std::string descType;

	bool empty() const {
		return rows == 0;
	}

	int type() const {
		return descType.compare("u") == 0 ? CV_8U : CV_32F;
	}

	size_t elemSize() const {
		return descType.compare("u") == 0 ? 1 : 4;
	}

};

/**
 * Opens a directory and saves all the files names onto a vector of strings, it
 * returns a status flag for reporting any error during the opening of the folder.
 *
 * @param folderPath - Path to the folder to be opened
 * @param files - Reference to a vector where all the files names will be saved
 */
void readFolder(const char* folderPath, std::vector<std::string>& files);

/**
 * Saves a list of strings to a plain text file, each element is saved to a new line.
 *
 * @param filename - The path to the file where to load the list from
 * @param list - The list where to save the loaded strings
 */
void saveList(const std::string& filename,
		const std::vector<std::string>& list);

/**
 * Loads a list of strings from a plain text file, each element is pushed as a new element.
 *
 * @param filename - The path to the file where to save the list
 * @param list - The list to be saved
 */
void loadList(const std::string& filename, std::vector<std::string>& list);

/**
 * Saves a set of features (keypoints and descriptors) onto a plain text
 * file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to save the features
 * @param keypoints - The keypoints to be saved
 * @param descriptors - The descriptors to be saved
 */
void saveFeatures(const std::string& filename,
		const std::vector<cv::KeyPoint>& keypoints, const cv::Mat& descriptors);

/**
 * Loads a set of features (keypoints and descriptors) from a plain text
 * file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to load the features from
 * @param keypoints - The keypoints where to save the loaded features
 * @param descriptors - The matrix of descriptors where to save the loaded features
 */
void loadFeatures(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors);

/**
 * Saves a set of descriptors onto a plain text file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to save the descriptors
 * @param descriptors - The descriptors to be saved
 */
void saveDescriptorsToYaml(const std::string& filename,
		const cv::Mat& descriptors);

/**
 * Loads a set of descriptors from a plain text file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to load the descriptors from
 * @param descriptors - The matrix where to save the loaded descriptors
 */
void loadDescriptorsFromYaml(const std::string& filename, cv::Mat& descriptors);

void saveDescriptors(const std::string& filename, const cv::Mat& descriptors);

void loadDescriptors(const std::string& filename, cv::Mat& descriptors);

/**
 * Saves a set of keypoints onto a plain text file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to save the keypoints
 * @param keypoints - The keypoints to be saved
 */
void saveKeypoints(const std::string& filename,
		const std::vector<cv::KeyPoint>& keypoints);

/**
 * Loads a set of keypoints from a plain text file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to load the keypoints from
 * @param keypoints - The list where to save the loaded keypoints
 */
void loadKeypoints(const std::string& filename,
		std::vector<cv::KeyPoint>& keypoints);

/**
 * Checks whether a file exists.
 *
 * @param filename - The name of the file to check
 * @return true if the file exists, false otherwise
 */
bool checkFileExist(const std::string& filename);

/**
 * Loads from a plain text file a list of strings and regions coordinates corresponding to
 * a set of queries.
 *
 * @param filePath - Path the file holding the list of queries and the coordinates of the region it determines
 * @param list - List holding the loaded queries
 */
void loadQueriesList(std::string& filePath, std::vector<Query>& list);

/**
 * Saves a set of descriptors in binary format onto a file stream using C++ STL.
 *
 * @param filename - The path to the file where to save the descriptors
 * @param descriptors - The descriptors to be saved
 */
void saveDescriptorsToBin(const std::string& filename,
		const cv::Mat& descriptors);

/**
 * Loads a set of descriptors from binary formatted file stream using C++ STL.
 *
 * @note This function does not do any memory deallocation. The invoker is
 * responsible for deallocating the matrix memory.
 *
 * @param filename - The path to the file where to load the descriptors from
 * @param descriptors - The matrix where to save the loaded descriptors
 */
void loadDescriptorsFromBin(const std::string& filename, cv::Mat& descriptors);

void loadDescriptorsFromZippedBin(const std::string& filename,
		cv::Mat& descriptors);

/**
 *
 * @param filename
 * @param descriptors
 * @param row - zero-based index
 */
void loadDescriptorsRowFromBin(const std::string& filename,
		cv::Mat& descriptors, int row);

void loadDescriptorsStats(std::string& filename, MatStats& stats);

void loadStatsFromZippedYaml(std::string& filename, MatStats& stats);

void loadStatsFromBin(std::string& filename, MatStats& stats);

} // namespace FileUtils

#endif
