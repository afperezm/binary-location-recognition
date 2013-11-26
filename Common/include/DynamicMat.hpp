/*
 * DynamicMat.hpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#ifndef DYNAMICMAT_HPP_
#define DYNAMICMAT_HPP_

#include <map>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <vector>
//#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

struct image {
	int imgIdx; // Index of the image it represents
	int startIdx; // (inclusive) Starting index in the merged descriptors matrix
	image() :
			imgIdx(-1), startIdx(-1) {
	}
};

static std::vector<image> DEFAULT_INDICES;
static std::vector<std::string> DEFAULT_FILENAMES;

class DynamicMat {

public:

	// Empty constructor
	DynamicMat();

	// Copy constructor
	DynamicMat(const DynamicMat& other);

	// Assignment operator
	DynamicMat& operator=(const DynamicMat& other);

	// Constructor
	DynamicMat(std::vector<std::string>& keysFilenames);

	// Destructor
	virtual ~DynamicMat();

	const std::vector<image>& getDescriptorsIndex() const {
		return m_descriptorsIndex;
	}

	const std::vector<std::string>& getDescriptorsFilenames() const {
		return m_descriptorsFilenames;
	}

	cv::Mat row(int descriptorIndex);

	int type() const;

	/**
	 * Returns true if the matrix is empty.
	 *
	 * @return true if rows counter is zero, false otherwise
	 */
	bool empty() const;

	/**
	 * Swaps the cache content by an empty vector of the same size.
	 */
	void clearCache();

	/**
	 * Returns the count of memory used by the matrices loaded in cache.
	 *
	 * @return memory count in Bytes
	 */
	int getMemoryCount() const {
		return m_memoryCount;
	}

private:
	static int computeUsedMemory(cv::Mat& descriptors) {
		return int(descriptors.rows * descriptors.cols * descriptors.elemSize());
	}

private:
	static const int MAX_MEM = 1075000000; // ~1GB

	std::vector<image> m_descriptorsIndex;
	std::vector<std::string> m_descriptorsFilenames;

	std::vector<cv::Mat> m_descriptorsCache;
	cv::Mat m_cachedMat;
	int m_cachedMatStartIdx;
	std::stack<int> m_cachingOrder;
//	std::queue<int> m_cachingOrder;

	int m_memoryCount = 0;
	int m_descriptorType = -1;

public:
	int rows = 0;
	int cols = 0;

};

#endif /* DYNAMICMAT_HPP_ */
