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
				0), m_descriptorsCache(NULL), rows(0), cols(0) {
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
	m_descriptorsCache = new std::map<int, cv::Mat>();
	m_cachingOrder = std::stack<int>();
	m_memoryCount = other.m_memoryCount;
	m_elemSize = other.elemSize();
	m_descriptorType = other.type();
	rows = other.rows;
	cols = other.cols;
}

// --------------------------------------------------------------------------

Mat::Mat(std::vector<std::string>& descriptorsFilenames) :
		m_descriptorsFilenames(descriptorsFilenames) {

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

	m_descriptorsCache = new std::map<int, cv::Mat>();
}

// --------------------------------------------------------------------------

Mat::~Mat() {
#if DYNMATVERBOSE
	printf("[DynamicMat] Destroying\n");
#endif
	delete m_descriptorsCache;
}

// --------------------------------------------------------------------------

Mat& Mat::operator=(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by assignment\n");
#endif

	m_capacity = other.getCapacity();
	m_imagesIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	m_descriptorsCache = new std::map<int, cv::Mat>();
	m_cachingOrder = std::stack<int>();
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

	if (descriptorIdx < 0 || descriptorIdx > int(m_imagesIndex.size())) {
		std::stringstream ss;
		ss << "[DynamicMat] Error while obtaining descriptor,"
				" the index should be in the range"
				" [0, " << m_imagesIndex.size() << ")";
		throw std::out_of_range(ss.str());
	}

	std::map<int, cv::Mat>::iterator it = m_descriptorsCache->find(
			descriptorIdx);

	if (it == m_descriptorsCache->end()) {

#if DYNMATVERBOSE
		printf("   NOT loaded in cache.\n");
#endif

		// Compute descriptor index relative to the descriptors matrix it belongs to
		int relIdx = descriptorIdx
				- m_descriptorsIndex[m_imagesIndex[descriptorIdx]];

		// Check whether cache is full
		if (m_memoryCount != 0 && m_memoryCount + rowMemorySize() > MAX_MEM) {

#if DYNMATVERBOSE
			printf("   Cache full, executing eviction policy.\n");
#endif

			/* Remove last added descriptor from the cache */
			// Obtain an iterator to it
			it = m_descriptorsCache->find(m_cachingOrder.top());
			// Check the iterator is valid
			CV_Assert(it != m_descriptorsCache->end());
			// Decrease memory counter
			m_memoryCount -= rowMemorySize();
			// Remove element from cache
			m_descriptorsCache->erase(it);
			// Pop its index from the stack
			m_cachingOrder.pop();
		}

		/* Add descriptor to the cache */
#if DYNMATVERBOSE
		printf("   Caching descriptor.\n");
#endif
		// Increase memory counter
		m_memoryCount += rowMemorySize();
		// Insert a new element
		it = m_descriptorsCache->insert(
				std::make_pair(descriptorIdx, cv::Mat(1, cols, type()))).first;
		FileUtils::loadDescriptorsRow(
				m_descriptorsFilenames[m_imagesIndex[descriptorIdx]],
				it->second, relIdx);
		// Push its index to the stack
		m_cachingOrder.push(descriptorIdx);
	} else {
#if DYNMATVERBOSE
		printf("   Loaded in cache.\n");
#endif
	}

	return it->second;
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
	m_descriptorsCache->clear();
	m_cachingOrder = std::stack<int>();
	m_memoryCount = 0;
}

} /* namespace vlr */
