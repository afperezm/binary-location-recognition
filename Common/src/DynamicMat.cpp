/*
 * DynamicMat.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#include <stdexcept>

#include <DynamicMat.hpp>
#include <FileUtils.hpp>

// --------------------------------------------------------------------------

DynamicMat::DynamicMat() :
		m_capacity(0), m_descriptorType(-1), m_descriptorsIndex(
				DEFAULT_INDICES), m_descriptorsFilenames(DEFAULT_FILENAMES), m_memoryCount(
				0), m_cachedMat(cv::Mat()), m_cachedMatStartIdx(-1), rows(0), cols(
				0) {
#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing empty\n");
#endif
}

// --------------------------------------------------------------------------

DynamicMat::DynamicMat(const DynamicMat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by copy\n");
#endif

	m_capacity = other.getCapacity();
	m_descriptorsIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	std::vector<cv::Mat>(m_descriptorsIndex.size(), cv::Mat()).swap(
			m_descriptorsCache);
	m_cachedMat = cv::Mat();
	m_cachedMatStartIdx = -1;
	m_cachingOrder = std::stack<int>();
	m_memoryCount = other.m_memoryCount;
	m_descriptorType = other.type();
	rows = other.rows;
	cols = other.cols;
}

// --------------------------------------------------------------------------

DynamicMat& DynamicMat::operator=(const DynamicMat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by assignment\n");
#endif

	m_descriptorsIndex = other.getDescriptorsIndex();
	m_descriptorsFilenames = other.getDescriptorsFilenames();
	std::vector<cv::Mat>(m_descriptorsIndex.size(), cv::Mat()).swap(
			m_descriptorsCache);
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

DynamicMat::DynamicMat(std::vector<std::string>& descriptorsFilenames) :
		m_descriptorsFilenames(descriptorsFilenames), m_cachedMatStartIdx(-1) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing using filenames\n");
#endif

	cv::Mat imgDescriptors;

	m_descriptorsIndex.clear();

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;
	size_t descElemSize = 0;

	double mytime = cv::getTickCount();

	// Initialize descriptors index
	for (std::string descriptorsFilename : descriptorsFilenames) {

		printf("[DynamicMat] Loading descriptors file [%d/%lu]\n", imgIdx + 1,
				descriptorsFilenames.size());

		// Clear descriptors matrix
		imgDescriptors = cv::Mat();

		// Load descriptors
		FileUtils::loadDescriptors(descriptorsFilename, imgDescriptors);

		if (imgDescriptors.empty() == false) {
			for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
				Image img;
				img.imgIdx = imgIdx;
				img.startIdx = descCount;
				m_descriptorsIndex.push_back(img);
			}
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
		imgIdx++;
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
	m_descriptorsCache.resize(m_descriptorsIndex.size(), cv::Mat());
}

// --------------------------------------------------------------------------

DynamicMat::~DynamicMat() {
#if DYNMATVERBOSE
	printf("[DynamicMat] Destroying\n");
#endif
}

// --------------------------------------------------------------------------

cv::Mat DynamicMat::row(int descriptorIdx) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Obtaining descriptor [%d]\n", descriptorIdx);
#endif

	if (descriptorIdx < 0 || descriptorIdx > int(m_descriptorsIndex.size())) {
		std::stringstream ss;
		ss << "[DynamicMat] Error while obtaining descriptor,"
				" the index should be in the range"
				" [0, " << m_descriptorsIndex.size() << ")";
		throw std::out_of_range(ss.str());
	}

	// Initialize descriptor
	cv::Mat descriptor = cv::Mat();

	std::vector<cv::Mat>::iterator it = m_descriptorsCache.begin()
			+ descriptorIdx;

	if ((*it).empty() == false) {
		// The descriptor is loaded in cache
		descriptor = *it;
	} else {
		// The descriptor is not loaded in cache

		// Load corresponding descriptors matrix if it isn't loaded
		if (m_cachedMatStartIdx == -1 || descriptorIdx < m_cachedMatStartIdx
				|| descriptorIdx >= m_cachedMatStartIdx + m_cachedMat.rows) {
			FileUtils::loadDescriptors(
					m_descriptorsFilenames[m_descriptorsIndex[descriptorIdx].imgIdx],
					m_cachedMat);
			m_cachedMatStartIdx = m_descriptorsIndex[descriptorIdx].startIdx;
		}

		// Compute descriptor index relative to the descriptors matrix it belongs to
		int relIdx = descriptorIdx - m_descriptorsIndex[descriptorIdx].startIdx;

		// Check relative descriptor index to be in range
		if (relIdx < 0 || relIdx > m_cachedMat.rows) {
			std::stringstream ss;
			ss << "Relative descriptor index [" << relIdx << "]"
					" should be in the range [0, " << m_cachedMat.rows << ")";
			throw std::out_of_range(ss.str());
		}

		// Obtain a reference to the descriptor
		descriptor = m_cachedMat.row(relIdx);

		// Check whether cache is full, if it does then pop the last added descriptor
		if (m_memoryCount != 0
				&& m_memoryCount + computeUsedMemory(descriptor) > MAX_MEM) {

#if DYNMATVERBOSE
			printf("[DynamicMat] Cache full, deleting last descriptor\n");
#endif

			// Remove descriptor from the cache
			// Obtain an iterator to it
			it = m_descriptorsCache.begin() + m_cachingOrder.top();
			// Decrease memory counter
			m_memoryCount -= computeUsedMemory(*it);
			// Dereference data
			*it = cv::Mat();
			// Pop its index from the stack
			m_cachingOrder.pop();
		}

		// Add descriptor to the cache
		// Obtain an iterator to it
		std::vector<cv::Mat>::iterator it = m_descriptorsCache.begin()
				+ descriptorIdx;
		// Increase memory counter
		m_memoryCount += computeUsedMemory(*it);
		// Copy data
		descriptor.copyTo(*it);
		// Push its index to the stack
		m_cachingOrder.push(descriptorIdx);
	}

	return *it;
}

// --------------------------------------------------------------------------

int DynamicMat::type() const {
	return m_descriptorType;
}

// --------------------------------------------------------------------------

bool DynamicMat::empty() const {
	return rows == 0;
}

void DynamicMat::clearCache() {
	std::vector<cv::Mat>(m_descriptorsIndex.size(), cv::Mat()).swap(
			m_descriptorsCache);
	m_cachingOrder = std::stack<int>();
	m_memoryCount = 0;
}
