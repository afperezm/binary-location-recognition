#ifndef __file_utils_h__
#define __file_utils_h__

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace FileUtils {

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
void saveDescriptors(const std::string& filename, const cv::Mat& descriptors);

/**
 * Loads a set of descriptors from a plain text file using OpenCV FileStorage API.
 *
 * @param filename - The path to the file where to load the descriptors from
 * @param descriptors - The matrix where to save the loaded descriptors
 */
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

} // namespace FileUtils

#endif
