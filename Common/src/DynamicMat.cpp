/*
 * DynamicMat.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

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

DynamicMat::DynamicMat(std::vector<std::string>& keysFilenames) {

	cv::Mat imgDescriptors;

	std::vector<image> descriptorsIndices;

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {

#if DYNMATVERBOSE
		printf("%d/%lu\n", imgIdx + 1, keysFilenames.size());
#endif

		// Initialize descriptors
		imgDescriptors = cv::Mat();

//		double mytime = cv::getTickCount();
		// Load keypoints and descriptors
		FileUtils::loadDescriptors(keyFileName, imgDescriptors);
//		mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency() * 1000.0;
//		printf("Loaded descriptors matrix in [%lf] ms\n", mytime);

		if (imgDescriptors.empty() == false) {
			for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
				image img;
				img.imgIdx = imgIdx;
				img.startIdx = descCount;
				descriptorsIndices.push_back(img);
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

	m_descriptorsIndices = descriptorsIndices;
	m_keysFilenames = keysFilenames;
	rows = descCount;
	cols = descLen;
	m_memoryCount = 0;
	m_descriptorType = descType;
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

	// Initialize descriptors
	cv::Mat imgDescriptors = cv::Mat();

	std::map<int, cv::Mat>::iterator it = descriptorCache.find(
			m_descriptorsIndices[descriptorIdx].imgIdx);

	if (it != descriptorCache.end()) {
		// The matrix is loaded in memory
		imgDescriptors = it->second;
	} else {
		// The matrix is not loaded in memory, then load it

		// Load corresponding descriptors file
		FileUtils::loadDescriptors(
				m_keysFilenames[m_descriptorsIndices[descriptorIdx].imgIdx],
				imgDescriptors);

		// Check buffer size. If full then pop the first element
		if (m_memoryCount != 0 && m_memoryCount + computeUsedMemory(imgDescriptors) > MAX_MEM) {

#if DYNMATVERBOSE
			fprintf(stdout, "[DynamicMat::row] Buffer full, deleting first matrix\n");
#endif

			// Find first element
			it = descriptorCache.find(addingOrder.front());
			// Decrease memory counter
			m_memoryCount -= computeUsedMemory(it->second);

			// Erase from the buffer the first added element
			descriptorCache.erase(it);
			// Pop its index from the queue
			addingOrder.pop();
		}

		// Add the descriptors matrix to the buffer
		descriptorCache.insert(
				std::pair<int, cv::Mat>(
						m_descriptorsIndices[descriptorIdx].imgIdx,
						imgDescriptors));
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
