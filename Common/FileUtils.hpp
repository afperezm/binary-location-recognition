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
void readFolder(const char* folderfolderPath, std::vector<std::string>& files);

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

} // namespace FileUtils

#endif
