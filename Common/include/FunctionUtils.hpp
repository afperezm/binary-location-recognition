/*
 * FunctionUtils.hpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#ifndef FUNCTIONUTILS_HPP_
#define FUNCTIONUTILS_HPP_

#include <vector>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

namespace FunctionUtils {

void printKeypoints(std::vector<cv::KeyPoint>& keypoints);

void printDescriptors(const cv::Mat& descriptors);

/**
 * Function for printing to standard output the parameters of <i>algorithm</i>
 *
 * @param algorithm A pointer to an object of type cv::Algorithm
 */
void printParams(cv::Ptr<cv::Algorithm> algorithm);

/**
 * Counts the number of bits equal to 1 in a specified number.
 *
 * @param i Number to count bits on
 * @return Number of bits equal to 1
 */
int NumberOfSetBits(int i);

int BinToDec(const cv::Mat& binRow);

std::vector<std::string> split(const std::string &ss, char delim);

void split(const std::string &s, char delim, std::vector<std::string> &tokens);

std::string basify(const std::string& s);

std::string parseLandmarkName(std::vector<std::string>::const_iterator fileName);

} // namespace FunctionUtils

#endif /* FUNCTIONUTILS_HPP_ */
