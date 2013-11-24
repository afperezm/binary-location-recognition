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
		m_descriptorsIndices(DEFAULT_INDICES), m_keysFilenames(DEFAULT_KEYS), rows(
				0), cols(0), m_memoryCount(0), m_descriptorType(-1) {

#if DYNMATVERBOSE
	fprintf(stdout, "Instantiation DynamicMat\n");
#endif

}

// --------------------------------------------------------------------------

DynamicMat::DynamicMat(std::vector<std::string>& keysFilenames) :
		m_keysFilenames(keysFilenames) {

	cv::Mat imgDescriptors;

	m_descriptorsIndices.clear();

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;

	double mytime = cv::getTickCount();

	for (std::string keyFileName : keysFilenames) {

		printf("Loading descriptor file [%d/%lu]\n", imgIdx + 1,
				keysFilenames.size());

		// Initialize descriptors
		imgDescriptors = cv::Mat();

		// Load keypoints and descriptors
		FileUtils::loadDescriptors(keyFileName, imgDescriptors);

		if (imgDescriptors.empty() == false) {
			for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
				image img;
				img.imgIdx = imgIdx;
				img.startIdx = descCount;
				m_descriptorsIndices.push_back(img);
			}
			// Increase descriptors counter
			descCount += imgDescriptors.rows;

			// If initialized check descriptors length
			// Recall all the descriptors must be the same length
			if (descLen != 0) {
				CV_Assert(descLen == imgDescriptors.cols);
			} else {
				descLen = imgDescriptors.cols;
			}
			// If initialized check descriptors type
			// Recall all the descriptors must be the same type
			if (descType != -1) {
				CV_Assert(descType == imgDescriptors.type());
			} else {
				descType = imgDescriptors.type();
			}
		}
		// Increase images counter
		imgIdx++;
	}

	mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency()
			* 1000;
	printf("Initialized descriptors index in [%lf] ms\n", mytime);

	rows = descCount;
	cols = descLen;
	m_memoryCount = 0;
	m_descriptorType = descType;
	m_descriptorCache.resize(imgIdx, cv::Mat());
}

// --------------------------------------------------------------------------

DynamicMat::~DynamicMat() {
#if DYNMATVERBOSE
	fprintf(stdout, "Destroying DynamicMat\n");
#endif
}

// --------------------------------------------------------------------------

DynamicMat::DynamicMat(const DynamicMat& other) {

#if DYNMATVERBOSE
	fprintf(stdout, "Copying DynamicMat\n");
#endif

#if DYNMATVERBOSE
	fprintf(stdout, "  Copying keypoint files names\n");
#endif
	m_keysFilenames = other.getKeysFilenames();

#if DYNMATVERBOSE
	fprintf(stdout, "  Copying descriptors indices\n");
#endif
	m_descriptorsIndices = other.getDescriptorsIndices();

#if DYNMATVERBOSE
	fprintf(stdout, "  Copying rows, columns and type information\n");
#endif

	rows = other.rows;
	cols = other.cols;
	m_descriptorType = other.type();
	m_memoryCount = other.m_memoryCount;

}

// --------------------------------------------------------------------------

DynamicMat& DynamicMat::operator =(const DynamicMat& other) {

#if DYNMATVERBOSE
	fprintf(stdout, "Assigning DynamicMat\n");
	fprintf(stdout, "  Copying key point files names\n");
#endif
	m_keysFilenames = other.getKeysFilenames();

#if DYNMATVERBOSE
	fprintf(stdout, "  Copying descriptors indices\n");
#endif
	m_descriptorsIndices = other.getDescriptorsIndices();

#if DYNMATVERBOSE
	fprintf(stdout, "  Copying rows, columns and type information\n");
#endif
	rows = other.rows;
	cols = other.cols;
	m_descriptorType = other.type();
	m_memoryCount = other.m_memoryCount;

	return *this;
}

// --------------------------------------------------------------------------

cv::Mat DynamicMat::row(int descriptorIdx) {

#if DYNMATVERBOSE
	fprintf(stdout, "[DynamicMat::row] Obtaining descriptor [%d]\n", descriptorIdx);
#endif

	if (descriptorIdx < 0 || descriptorIdx > int(m_descriptorsIndices.size())) {
		std::stringstream ss;
		ss << "[DynamicMat::row] Descriptor index should be in the range"
				" [0, " << m_descriptorsIndices.size() << "]";
		throw std::out_of_range(ss.str());
	}

	// Initialize descriptors
	cv::Mat imgDescriptors = cv::Mat();

	std::vector<cv::Mat>::iterator it = m_descriptorCache.begin()
			+ m_descriptorsIndices[descriptorIdx].imgIdx;

	if ((*it).empty() == false) {
		// The matrix is loaded in memory
		imgDescriptors = *it;
	} else {
		// The matrix is not loaded in memory, then load it

		// Load corresponding descriptors file
		FileUtils::loadDescriptors(
				m_keysFilenames[m_descriptorsIndices[descriptorIdx].imgIdx],
				imgDescriptors);

		// Check buffer size. If full then pop the first element
		if (m_memoryCount != 0
				&& m_memoryCount + computeUsedMemory(imgDescriptors)
						> MAX_MEM) {

#if DYNMATVERBOSE
			fprintf(stdout, "[DynamicMat::row] Buffer full, deleting first matrix\n");
#endif

			// Find first added element
			it = m_descriptorCache.begin() + addingOrder.front();
			// Decrease memory counter
			m_memoryCount -= computeUsedMemory(*it);

			// Erase from the buffer the first added element
			*it = cv::Mat();
			// Pop its index from the queue
			addingOrder.pop();
		}

		// Add matrix to the cache
		it = m_descriptorCache.begin()
				+ m_descriptorsIndices[descriptorIdx].imgIdx;
		m_descriptorCache.insert(it, imgDescriptors);
		addingOrder.push(m_descriptorsIndices[descriptorIdx].imgIdx);

		// Increase memory counter
		m_memoryCount += computeUsedMemory(imgDescriptors);
	}

	// Index relative to the matrix of descriptors it belongs to
	int relDescIdx = descriptorIdx
			- m_descriptorsIndices[descriptorIdx].startIdx;

	return imgDescriptors.row(relDescIdx);
}

// --------------------------------------------------------------------------

int DynamicMat::type() const {
	return m_descriptorType;
}

// --------------------------------------------------------------------------

bool DynamicMat::empty() const {
	return rows == 0;
}

void DynamicMat::release() {
	std::fill(m_descriptorCache.begin(),
			m_descriptorCache.begin() + m_descriptorCache.size(), cv::Mat());
}
