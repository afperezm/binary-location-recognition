/*
 * DynamicMat.hpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#ifndef DYNAMICMAT_HPP_
#define DYNAMICMAT_HPP_

#include <map>
#include <math.h>
#include <sstream>
#include <stack>
#include <stdio.h>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

struct Image {
	int imgIdx; // Index of the image it represents
	int startIdx; // (inclusive) Starting index in the merged descriptors matrix
	Image() :
			imgIdx(-1), startIdx(-1) {
	}
};

static std::vector<Image> DEFAULT_INDICES;
static std::vector<std::string> DEFAULT_FILENAMES;

class DynamicMat {

private:

	static const int MAX_MEM = 1075000000; // ~1GB

	double m_capacity = 0.0;
	int m_descriptorType = -1;
	std::vector<Image> m_descriptorsIndex;
	std::vector<std::string> m_descriptorsFilenames;

	/** Attributes of the cache **/
	int m_memoryCount = 0;
	cv::Mat m_cachedMat;
	int m_cachedMatStartIdx;
	std::stack<int> m_cachingOrder;
	std::vector<cv::Mat> m_descriptorsCache;

public:

	int rows = 0;
	int cols = 0;

public:

	/**
	 * Class empty constructor.
	 */
	DynamicMat();

	/**
	 * Class constructor.
	 *
	 * @param keysFilenames - Reference to a vector of descriptor filenames
	 */
	DynamicMat(std::vector<std::string>& keysFilenames);

	/**
	 * Copy constructor.
	 *
	 * @param other - Reference to an instance where to copy properties from
	 */
	DynamicMat(const DynamicMat& other);

	/**
	 * Class destroyer.
	 */
	virtual ~DynamicMat();

	/**
	 * Assignment operator.
	 *
	 * @param other - Reference to an instance where to copy properties from
	 * @return reference to the new instance
	 */
	DynamicMat& operator=(const DynamicMat& other);

	/**
	 * Retrieves the requested descriptor by obtaining it from cache
	 * and if necessary load the associated descriptor.
	 *
	 * @param descriptorIndex - Index of the descriptor to retrieve
	 * @return requested descriptor
	 */
	cv::Mat row(int descriptorIndex);

	/**
	 * Returns type of descriptors held by the virtual big descriptors matrix.
	 *
	 * @return descriptors type
	 */
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
	 * Returns the maximum number of descriptors that can be stored
	 * given the maximum size of the cache
	 *
	 * @return cache capacity in number of descriptors
	 */
	double getCapacity() const {
		return m_capacity;
	}

	/** Getters **/

	/**
	 * Returns a reference to the index descriptors holding the
	 * index of image it belongs to and the starting index of the
	 * virtual big descriptors matrix.
	 *
	 * @return index of descriptors
	 */
	const std::vector<Image>& getDescriptorsIndex() const {
		return m_descriptorsIndex;
	}

	/**
	 * Returns a reference to the vector of descriptors filenames.
	 *
	 * @return descriptors filenames
	 */
	const std::vector<std::string>& getDescriptorsFilenames() const {
		return m_descriptorsFilenames;
	}

	/**
	 * Returns the count of memory used by the matrices loaded in cache.
	 *
	 * @return memory count in Bytes
	 */
	int getMemoryCount() const {
		return m_memoryCount;
	}

private:

	/**
	 * Computes used memory by the given descriptors matrix
	 *
	 * @param descriptors - a reference to a matrix containing descriptors
	 * @return number of bytes used
	 */
	static int computeUsedMemory(cv::Mat& descriptors) {
		return int(descriptors.rows * descriptors.cols * descriptors.elemSize());
	}

};

#endif /* DYNAMICMAT_HPP_ */
