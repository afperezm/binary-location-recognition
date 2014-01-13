/*
 * matching.cpp
 *
 *  Created on: Dec 18, 2013
 *      Author: andresf
 */

#include <matching.hpp>

void matchKeypoints(const cv::Ptr<bfeat::DirectIndex> directIndex1, int idImg1,
		const std::vector<cv::KeyPoint>& keypoints1,
		std::vector<cv::Point2f>& matchedPoints1,
		const cv::Ptr<bfeat::DirectIndex> directIndex2, int idImg2,
		const std::vector<cv::KeyPoint>& keypoints2,
		std::vector<cv::Point2f>& matchedPoints2,
		std::vector<cv::DMatch>& matches1to2, double proximityThreshold) {

	// Clean up variables received as arguments
	matchedPoints1.clear();
	matchedPoints2.clear();
	matches1to2.clear();

	// Lookup query and database images in the index and get nodes list
	bfeat::TreeNode nodes1 = directIndex1->lookUpImg(idImg1);
	bfeat::TreeNode nodes2 = directIndex2->lookUpImg(idImg2);

	// Declare and initialize iterators to the maps to intersect
	typename bfeat::TreeNode::const_iterator it1 = nodes1.begin();
	typename bfeat::TreeNode::const_iterator it2 = nodes2.begin();

	cv::Point2f point1, point2;

	double kptsDist;

	// Intersect nodes maps, solution taken from:
	// http://stackoverflow.com/questions/3772664/intersection-of-two-stl-maps
	while (it1 != nodes1.end() && it2 != nodes2.end()) {
		if (it1->first < it2->first) {
			++it1;
		} else if (it2->first < it1->first) {
			++it2;
		} else {
			// Match together all query keypoints vs candidate keypoints of an intersected node
			for (int i1 : it1->second) {
				for (int i2 : it2->second) {

					// Assert that feature ids stored in the direct index are in range
					CV_Assert(
							i1 >= 0
									&& i1
											< static_cast<int>(keypoints1.size()));
					CV_Assert(
							i2 >= 0
									&& i2
											< static_cast<int>(keypoints2.size()));

					// Extract point
					point1 = keypoints1[i1].pt;
					point2 = keypoints2[i2].pt;

					kptsDist = cv::norm(
							cv::Point(point1.x, point1.y)
									- cv::Point(point2.x, point2.y));

					// Apply a proximity threshold
					if (kptsDist > proximityThreshold) {
						continue;
					}

					// Add points to vectors of matched
					matchedPoints1.push_back(point1);
					matchedPoints2.push_back(point2);

					cv::DMatch match = cv::DMatch(i1, i2,
							cv::norm(point1 - point2));
					// Set pair as a match
					matches1to2.push_back(match);
				}
			}
			++it1, ++it2;
		}
	}

}

void matchKeypoints(std::vector<cv::KeyPoint>& keypoints1,
		cv::Mat& descriptors1, std::vector<cv::KeyPoint>& keypoints2,
		cv::Mat& descriptors2, std::vector<cv::DMatch>& matches1to2,
		int topKeypoints, double ratioThreshold) {

	// Clean up non constant variables received as parameters
	matches1to2.clear();

	CV_Assert(descriptors1.cols == descriptors2.cols);
	CV_Assert(descriptors1.type() == descriptors2.type());
	CV_Assert(int(keypoints1.size()) == descriptors1.rows);
	CV_Assert(int(keypoints2.size()) == descriptors2.rows);

	if (descriptors1.type() == CV_8U) {
		_matchKeypointsImpl<uchar, cv::Hamming>(keypoints1, descriptors1,
				keypoints2, descriptors2, matches1to2, ratioThreshold);
	} else {
		_matchKeypointsImpl<float, cv::L2<float> >(keypoints1, descriptors1,
				keypoints2, descriptors2, matches1to2, ratioThreshold);
	}

}

template<class TDescriptor, class Distance>
void _matchKeypointsImpl(const std::vector<cv::KeyPoint>& keypoints1,
		const cv::Mat& descriptors1,
		const std::vector<cv::KeyPoint>& keypoints2,
		const cv::Mat& descriptors2, std::vector<cv::DMatch>& matches1to2,
		const double& ratioThreshold) {

	Distance distance = Distance();

	typedef typename Distance::ResultType DistanceType;

	double score;
	DistanceType descsDist, dBest, dSecond;
	int idBest = -1, idSecondBest = -1;

	// knn(2): for each descriptor in 1 find the best two descriptors in 2
	for (size_t i = 0; i < keypoints1.size(); i++) {

		descsDist = distance((TDescriptor*) descriptors1.row(i).data,
				(TDescriptor*) descriptors2.row(0).data, descriptors1.cols);
		dBest = descsDist;
		dSecond = dBest;
		idBest = 0;

		for (size_t j = 1; j < keypoints2.size(); j++) {

			descsDist = distance((TDescriptor*) descriptors1.row(i).data,
					(TDescriptor*) descriptors2.row(j).data, descriptors1.cols);

			if (descsDist < dBest) {
				dSecond = dBest;
				dBest = descsDist;
				idBest = int(j);
			} else if (descsDist < dSecond) {
				dSecond = descsDist;
				idSecondBest = int(j);
			}

		}

		score = double(dBest) / double(dSecond);

		CV_Assert(score <= 1.0);

		// Reject all matches in which the distance ratio is greater than 0.8
		// this eliminates 90% of the false matches while discarding less than 5% of the correct matches
		if (score > ratioThreshold) {
			continue;
		}

		// the nearest point (stored at the trainIdx position) to some other point (stored at the queryIdxposition)
		cv::DMatch match = cv::DMatch(i, idBest,
				distance((TDescriptor*) descriptors1.row(i).data,
						(TDescriptor*) descriptors2.row(idBest).data,
						descriptors1.cols));

		// Set pair as a match
		matches1to2.push_back(match);

	}

}

template<typename KeyType, typename LeftValue, typename RightValue>
std::map<KeyType, std::pair<LeftValue, RightValue> > intersectMaps(
		const std::map<KeyType, LeftValue> & left,
		const std::map<KeyType, RightValue> & right) {
	std::map<KeyType, std::pair<LeftValue, RightValue> > result;
	typename std::map<KeyType, LeftValue>::const_iterator il = left.begin();
	typename std::map<KeyType, RightValue>::const_iterator ir = right.begin();
	while (il != left.end() && ir != right.end()) {
		if (il->first < ir->first)
			++il;
		else if (ir->first < il->first)
			++ir;
		else {
			result.insert(
					std::make_pair(il->first,
							std::make_pair(il->second, ir->second)));
			++il;
			++ir;
		}
	}
	return result;
}

void filterFeatures(std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors,
		int top) {

	CV_Assert(int(keypoints.size()) == descriptors.rows);

	if (top < 0) {
		// Do nothing
		return;
	}

	std::vector<cv::KeyPoint> topKeypoints;
	cv::Mat topDescriptors;

	// Sort key-points according to the response
	std::vector<size_t> indices;
	sortKptsByResponse(keypoints, indices);

	CV_Assert(keypoints.size() == indices.size());

	top = MIN(top, int(indices.size()));

	// Create temporary descriptor matrix where to save the top features
	topDescriptors.create(0, descriptors.cols, descriptors.type());
	topDescriptors.reserve(top);

	// Copy top features to temporary key-points vector and descriptors matrix
	for (int i = 0; i < top; ++i) {
		CV_Assert(indices[i] < keypoints.size());
		CV_Assert(int(indices[i]) < descriptors.rows);
		topKeypoints.push_back(keypoints[indices[i]]);
		topDescriptors.push_back(descriptors.row(indices[i]));
	}

	CV_Assert(int(topKeypoints.size()) == topDescriptors.rows);

	// Copy back temporary key-points vector and descriptors matrix
	keypoints = topKeypoints;
	descriptors = topDescriptors.clone();

}

void sortKptsByResponse(const std::vector<cv::KeyPoint>& values,
		std::vector<size_t>& indices) {

	indices.clear();
	indices.resize(values.size());

	for (size_t i = 0; i < indices.size(); ++i) {
		indices[i] = i;
	}

	std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
		return values[a].response > values[b].response;
	});

}

