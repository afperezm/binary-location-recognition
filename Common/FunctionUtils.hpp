/*
 * FunctionUtils.hpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#ifndef FUNCTIONUTILS_HPP_
#define FUNCTIONUTILS_HPP_

#include <vector>

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

} // namespace FunctionUtils

#endif /* FUNCTIONUTILS_HPP_ */
