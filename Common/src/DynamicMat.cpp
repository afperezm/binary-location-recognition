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
		m_capacity(0.0), m_descriptorType(-1), m_imagesIndex(DEFAULT_INDICES), m_descriptorsFilenames(
				DEFAULT_FILENAMES), m_memoryCount(0), m_cachedMat(cv::Mat()), m_cachedMatStartIdx(
				-1), m_evictionPolicyActive(true), rows(0), cols(0) {
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
	m_descriptorsCache.clear();
	m_cachedMat = cv::Mat();
	m_cachedMatStartIdx = -1;
	m_cachingOrder = std::stack<int>();
	m_memoryCount = other.m_memoryCount;
	m_descriptorType = other.type();
	rows = other.rows;
	cols = other.cols;
	m_evictionPolicyActive = other.isEvictionPolicyActive();
}

// --------------------------------------------------------------------------

Mat::Mat(std::vector<std::string>& descriptorsFilenames) :
		m_descriptorsFilenames(descriptorsFilenames), m_cachedMatStartIdx(-1), m_evictionPolicyActive(
				true) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing using filenames\n");
#endif

	FileUtils::MatStats imgDescriptors;

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
		FileUtils::loadDescriptorsStats(descriptorsFilename, imgDescriptors);

		// Increase counter if descriptors matrix is not empty
		if (imgDescriptors.empty() == false) {
			m_imagesIndex.insert(m_imagesIndex.end(), imgDescriptors.rows,
					imgIdx);
			m_descriptorsIndex[imgIdx] = descCount;
			// Increase descriptors counter
			descCount += imgDescriptors.rows;

			// If initialized check descriptors length
			if (descLen != 0) {
				// Recall that all descriptors must be of the same length
				CV_Assert(descLen == imgDescriptors.cols);
			} else {
				descLen = imgDescriptors.cols;
			}
			// If initialized check descriptors type
			if (descType != -1) {
				// Recall that all descriptors must be of the same type
				CV_Assert(descType == imgDescriptors.type());
			} else {
				descType = imgDescriptors.type();
			}
			// If initialized check descriptors element size
			if (descElemSize != 0) {
				// Recall that all descriptors must have the same element size
				CV_Assert(descElemSize == imgDescriptors.elemSize());
			} else {
				descElemSize = imgDescriptors.elemSize();
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

	CV_Assert(cols > 0 && descElemSize > 0);

	m_capacity = floor(double(MAX_MEM / (cols * descElemSize)));

	m_descriptorsCache.clear();
}

// --------------------------------------------------------------------------

Mat::~Mat() {
#if DYNMATVERBOSE
	printf("[DynamicMat] Destroying\n");
#endif
	clearCache();
}

// --------------------------------------------------------------------------

Mat& Mat::operator=(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by assignment\n");
#endif

	m_imagesIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	m_descriptorsCache.clear();
	m_cachedMat = cv::Mat();
	m_cachedMatStartIdx = -1;
	m_cachingOrder = std::stack<int>();
	m_memoryCount = other.m_memoryCount;
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

	std::map<int, cv::Mat>::iterator it = m_descriptorsCache.find(
			descriptorIdx);

	if (it == m_descriptorsCache.end()) {

#if DYNMATVERBOSE
		printf("   NOT loaded in cache.\n");
#endif

		// Initialize descriptor
		cv::Mat descriptor = cv::Mat();

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
		if (relIdx < 0 || relIdx > m_cachedMat.rows) {
			std::stringstream ss;
			ss << "Relative descriptor index [" << relIdx << "]"
					" should be in the range [0, " << m_cachedMat.rows << ")";
			throw std::out_of_range(ss.str());
		}

		// Obtain a reference to the descriptor
		descriptor = m_cachedMat.row(relIdx);

		// Check whether cache is full
		if (m_memoryCount != 0
				&& m_memoryCount + computeUsedMemory(descriptor) > MAX_MEM) {

			/* Return row of cached matrix */
			if (m_evictionPolicyActive == false) {
#if DYNMATVERBOSE
				printf("   Cache full, skipping eviction policy.\n");
#endif
				return descriptor;
			}

#if DYNMATVERBOSE
			printf("   Cache full, executing eviction policy.\n");
#endif

			/* Remove last added descriptor from the cache */
			// Obtain an iterator to it
			it = m_descriptorsCache.find(m_cachingOrder.top());
			// Check the iterator is valid
			CV_Assert(it != m_descriptorsCache.end());
			// Decrease memory counter
			m_memoryCount -= computeUsedMemory(it->second);
			// Release and dereference data
			it->second.release();
			it->second = cv::Mat();
			// Remove element from cache
			m_descriptorsCache.erase(it);
			// Pop its index from the stack
			m_cachingOrder.pop();
		}

		/* Add descriptor to the cache */
#if DYNMATVERBOSE
			printf("   Caching descriptor.\n");
#endif
		// Increase memory counter
		m_memoryCount += computeUsedMemory(descriptor);
		// Insert a new element
		it =
				m_descriptorsCache.insert(
						std::make_pair(descriptorIdx, cv::Mat())).first;
		// Copy data
		descriptor.copyTo(m_descriptorsCache.at(descriptorIdx));
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

bool Mat::empty() const {
	return rows == 0;
}

void Mat::clearCache() {
	m_descriptorsCache.clear();
	m_cachingOrder = std::stack<int>();
	m_memoryCount = 0;
}

} /* namespace vlr */
