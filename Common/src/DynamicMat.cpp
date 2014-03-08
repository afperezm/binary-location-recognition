/*
 * DynamicMat.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#include <stdexcept>

#include <DynamicMat.hpp>
#include <FileUtils.hpp>

namespace vlr {

Mat::Mat() :
		m_capacity(0.0), m_descriptorType(-1), m_elemSize(0), m_imagesIndex(
				DEFAULT_INDICES), m_descriptorsFilenames(DEFAULT_FILENAMES), m_memoryCount(
				0), m_cachedMat(cv::Mat()), m_cachedMatStartIdx(-1), m_cachingOrder(
				NULL), m_cacheIndex(NULL), m_cache(NULL), rows(0), cols(0) {
#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing empty\n");
#endif
}

// --------------------------------------------------------------------------

Mat::Mat(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by copy\n");
#endif

	m_capacity = other.getCapacity();
	m_imagesIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	m_cachedMat = cv::Mat();
	m_cachedMatStartIdx = -1;
	m_cachingOrder = new std::stack<int>();
	m_cacheIndex = new std::vector<int>();
	m_cache = new cv::Mat(0, cols, m_descriptorType);
	m_memoryCount = other.m_memoryCount;
	m_elemSize = other.elemSize();
	m_descriptorType = other.type();
	rows = other.rows;
	cols = other.cols;
}

// --------------------------------------------------------------------------

Mat::Mat(std::vector<std::string>& descriptorsFilenames) :
		m_descriptorsFilenames(descriptorsFilenames), m_cachedMatStartIdx(-1) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing using filenames\n");
#endif

	FileUtils::MatStats descriptorsStats;

	m_imagesIndex.clear();
	m_descriptorsIndex.clear();
	m_descriptorsIndex.resize(descriptorsFilenames.size(), 0);

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;
	size_t descElemSize = 0;

	double mytime = cv::getTickCount();

	// Initialize descriptors index
	for (std::string& descriptorsFilename : descriptorsFilenames) {

		printf("[DynamicMat] Loading descriptors file [%04d/%04lu]\n",
				imgIdx + 1, descriptorsFilenames.size());

		// Load descriptors
		FileUtils::loadDescriptorsStats(descriptorsFilename, descriptorsStats);

		// Increase counter if descriptors matrix is not empty
		if (descriptorsStats.empty() == false) {
			m_imagesIndex.insert(m_imagesIndex.end(), descriptorsStats.rows,
					imgIdx);
			m_descriptorsIndex[imgIdx] = descCount;
			// Increase descriptors counter
			descCount += descriptorsStats.rows;

			// If initialized check descriptors length
			if (descLen != 0) {
				// Recall that all descriptors must be of the same length
				CV_Assert(descLen == descriptorsStats.cols);
			} else {
				descLen = descriptorsStats.cols;
			}
			// If initialized check descriptors type
			if (descType != -1) {
				// Recall that all descriptors must be of the same type
				CV_Assert(descType == descriptorsStats.type());
			} else {
				descType = descriptorsStats.type();
			}
			// If initialized check descriptors element size
			if (descElemSize != 0) {
				// Recall that all descriptors must have the same element size
				CV_Assert(descElemSize == descriptorsStats.elemSize());
			} else {
				descElemSize = descriptorsStats.elemSize();
			}
		}

		// Increase images counter
		++imgIdx;
	}

	mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency()
			* 1000;
#if DYNMATVERBOSE
	printf("[DynamicMat] Initialized descriptors index in [%lf] ms\n", mytime);
#endif

	rows = descCount;
	cols = descLen;
	m_memoryCount = 0;
	m_descriptorType = descType;
	m_elemSize = descElemSize;

	CV_Assert(cols > 0 && descElemSize > 0);

	m_capacity = floor(double(MAX_MEM / (cols * descElemSize)));

	m_cachingOrder = new std::stack<int>();
	m_cacheIndex = new std::vector<int>(rows, -1);
	m_cache = new cv::Mat(0, cols, m_descriptorType);
}

// --------------------------------------------------------------------------

Mat::~Mat() {
#if DYNMATVERBOSE
	printf("[DynamicMat] Destroying\n");
#endif
	delete m_cachingOrder;
	delete m_cacheIndex;
	delete m_cache;
}

// --------------------------------------------------------------------------

Mat& Mat::operator=(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by assignment\n");
#endif

	m_capacity = other.getCapacity();
	m_imagesIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	m_cachedMat = cv::Mat();
	m_cachingOrder = new std::stack<int>();
	m_cacheIndex = new std::vector<int>();
	m_cache = new cv::Mat(0, cols, m_descriptorType);
	m_memoryCount = other.m_memoryCount;
	m_elemSize = other.elemSize();
	m_descriptorType = other.type();
	rows = other.rows;
	cols = other.cols;

	return *this;
}

// --------------------------------------------------------------------------

cv::Mat Mat::row(int descriptorIdx) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Obtaining descriptor [%d]\n", descriptorIdx);
#endif

	if (descriptorIdx < 0 || descriptorIdx > rows) {
		std::stringstream ss;
		ss << "[DynamicMat] Error while obtaining descriptor,"
				" the index should be in the range"
				" [0, " << rows << ")";
		throw std::out_of_range(ss.str());
	}

	std::vector<int>::iterator it = m_cacheIndex->begin() + descriptorIdx;

	// Iterator is invalid if cache index doesn't contain descriptor index
	CV_Assert(it != m_cacheIndex->end());

	if (*it == -1) {

#if DYNMATVERBOSE
		printf("   NOT loaded in cache.\n");
#endif

		// Load corresponding descriptors matrix if it isn't loaded
		if (m_cachedMatStartIdx == -1 || descriptorIdx < m_cachedMatStartIdx
				|| descriptorIdx >= m_cachedMatStartIdx + m_cachedMat.rows) {
			m_cachedMat.release();
			FileUtils::loadDescriptors(
					m_descriptorsFilenames[m_imagesIndex[descriptorIdx]],
					m_cachedMat);
			m_cachedMatStartIdx =
					m_descriptorsIndex[m_imagesIndex[descriptorIdx]];
		}

		// Compute descriptor index relative to the descriptors matrix it belongs to
		int relIdx = descriptorIdx
				- m_descriptorsIndex[m_imagesIndex[descriptorIdx]];

		// Check relative descriptor index to be in range
		CV_Assert(relIdx >= 0 || relIdx < m_cachedMat.rows);

		// Obtain a reference to the descriptor
		// descriptor = m_cachedMat.row(relIdx);

		// Check whether cache is full
		if (m_memoryCount != 0 && m_memoryCount + rowMemorySize() > MAX_MEM) {

#if DYNMATVERBOSE
			printf("   Cache full, executing eviction policy.\n");
#endif

			/* Remove last added descriptor from the cache */
			// Obtain an iterator to it
			it = m_cacheIndex->begin() + m_cachingOrder->top();
			// Check the iterator is valid
			CV_Assert(it != m_cacheIndex->end());
			// Decrease memory counter
			m_memoryCount -= rowMemorySize();
			// Remove element from cache
			m_cacheIndex->at(*it) = -1;
			m_cache->pop_back();
			// Pop its index from the stack
			m_cachingOrder->pop();
		}

		/* Add descriptor to the cache */
#if DYNMATVERBOSE
		printf("   Caching descriptor.\n");
#endif
		// Increase memory counter
		m_memoryCount += rowMemorySize();
		// Insert a new element
		m_cacheIndex->at(descriptorIdx) = m_cache->rows;
		m_cache->push_back(cv::Mat(1, cols, type()));
		it = m_cacheIndex->begin() + descriptorIdx;
		// Check the iterator is valid
		CV_Assert(it != m_cacheIndex->end());
		// Load descriptor
		m_cachedMat.row(relIdx).copyTo(m_cache->row(*it));
		// Push its index to the stack
		m_cachingOrder->push(descriptorIdx);
	} else {
#if DYNMATVERBOSE
		printf("   Loaded in cache.\n");
#endif
	}

	return m_cache->row(*it);
}

// --------------------------------------------------------------------------

int Mat::type() const {
	return m_descriptorType;
}

// --------------------------------------------------------------------------

size_t Mat::elemSize() const {
	return m_elemSize;
}

// --------------------------------------------------------------------------

bool Mat::empty() const {
	return rows == 0;
}

void Mat::clearCache() {
	m_memoryCount = 0;
	while (m_cachingOrder->empty() == false) {
		m_cachingOrder->pop();
	}
//	while (m_cache->empty() == false) {
//		m_cache->pop_back();
//	}
	m_cache->release();
}

} /* namespace vlr */
