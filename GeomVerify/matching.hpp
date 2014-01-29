/*
 * matching.hpp
 *
 *  Created on: Dec 18, 2013
 *      Author: andresf
 */

#ifndef MATCHING_HPP_
#define MATCHING_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <DirectIndex.hpp>

/**
 *
 * @param directIndex1
 * @param idImg1
 * @param keypoints1
 * @param matchedPoints1
 * @param directIndex2
 * @param idImg2
 * @param keypoints2
 * @param matchedPoints2
 * @param matches1to2
 * @param proximityThreshold
 */
void matchKeypoints(const cv::Ptr<vlr::DirectIndex> directIndex1, int idImg1,
		const std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::Point2f>& matchedPoints1,
		const cv::Ptr<vlr::DirectIndex> directIndex2, int idImg2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2,
		std::vector<cv::DMatch>& matches1to2, double proximityThreshold);

/**
 *
 * @param keypoints1
 * @param descriptors1
 * @param keypoints2
 * @param descriptors2
 * @param matches1to2
 * @param topKeypoints
 * @param ratioThreshold
 */
void matchKeypoints(std::vector<cv::KeyPoint>& keypoints1,
		cv::Mat& descriptors1, std::vector<cv::KeyPoint>& keypoints2,
		cv::Mat& descriptors2, std::vector<cv::DMatch>& matches1to2,
		int topKeypoints = 500, double ratioThreshold = 0.8);

/**
 *
 * @param keypoints1
 * @param descriptors1
 * @param keypoints2
 * @param descriptors2
 * @param matches1to2
 * @param ratioThreshold
 */
template<class TDescriptor, class Distance>
void _matchKeypointsImpl(const std::vector<cv::KeyPoint>& keypoints1,
		const cv::Mat& descriptors1,
		const std::vector<cv::KeyPoint>& keypoints2,
		const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches1to2,
		const double& ratioThreshold);

/**
 *
 * @param left
 * @param right
 * @return
 */
template<typename KeyType, typename LeftValue, typename RightValue>
std::map<KeyType, std::pair<LeftValue, RightValue> > intersectMaps(
		const std::map<KeyType, LeftValue> & left,
		const std::map<KeyType, RightValue> & right);

/**
 * Sorts a vector in ascending/descending order by keeping track of the indices.
 *
 * @param values
 * @param indices
 *
 * @return
 */
void sortKptsByResponse(std::vector<cv::KeyPoint> const& values,
		std::vector<size_t>& indices);

/**
 * Filters out features in order to keep the ones with higher key-point response.
 *
 * @param keypoints - The vector of key-points corresponding to the features to filter
 * @param descriptors - The matrix of descriptors corresponding to the features to filter
 * @param topKeypoints - Top number of features to keep
 */
void filterFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
		int top);

#endif /* MATCHING_HPP_ */
