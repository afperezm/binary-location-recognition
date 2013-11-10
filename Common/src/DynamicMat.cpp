/*
 * DynamicMat.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#include <DynamicMat.hpp>
#include <FileUtils.hpp>

// --------------------------------------------------------------------------

DynamicMat::DynamicMat(std::vector<image>& descriptorsIndices,
		std::vector<std::string>& keysFilenames, int descriptorCount,
		int descriptorLength, int descriptorType) :
		m_descriptorsIndices(descriptorsIndices), m_keysFilenames(
				keysFilenames), rows(descriptorCount), cols(descriptorLength), m_memoryCounter(
				0), m_descriptorType(descriptorType) {

#if DYNMATVERBOSE
	fprintf(stdout, "Instantiation DynamicMat\n");
#endif

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
	m_memoryCounter = other.m_memoryCounter;

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
	m_memoryCounter = other.m_memoryCounter;

	return *this;
}

// --------------------------------------------------------------------------

cv::Mat DynamicMat::row(int descriptorIdx) {

#if DYNMATVERBOSE
	fprintf(stdout, "[DynamicMat::row] Obtaining descriptor [%d]\n", descriptorIdx);
#endif

	// Initialize keypoints and descriptors
	std::vector<cv::KeyPoint> imgKeypoints;
	cv::Mat imgDescriptors = cv::Mat();

	std::map<int, cv::Mat>::iterator it = descBuffer.find(
			m_descriptorsIndices[descriptorIdx].imgIdx);

	if (it != descBuffer.end()) {
		// The matrix is loaded in memory
		imgDescriptors = it->second;
	} else {
		// The matrix is not loaded in memory, then load it

		// Load corresponding descriptors file
		FileUtils::loadFeatures(
				m_keysFilenames[m_descriptorsIndices[descriptorIdx].imgIdx],
				imgKeypoints, imgDescriptors);

		// Check buffer size, if full then pop the first element
		if (m_memoryCounter > MAX_MEM) {
#if DYNMATVERBOSE
			fprintf(stdout, "[DynamicMat::row] Buffer full, deleting first matrix\n");
#endif
			// Find first element
			it = descBuffer.find(addingOrder.front());
			// Decrease memory counter
			if (imgDescriptors.type() == CV_8U) {
				m_memoryCounter -=
						(int) (it->second.rows * cols * sizeof(uchar));
			} else {
				m_memoryCounter -=
						(int) (it->second.rows * cols * sizeof(float));
			}
			// Erase from the buffer the first added element
			descBuffer.erase(it);
			// Pop its index from the queue
			addingOrder.pop();
		}

		// Add the descriptors matrix to the buffer
		descBuffer.insert(
				std::pair<int, cv::Mat>(
						m_descriptorsIndices[descriptorIdx].imgIdx,
						imgDescriptors));
		addingOrder.push(m_descriptorsIndices[descriptorIdx].imgIdx);
		// Increase memory counter
		if (imgDescriptors.type() == CV_8U) {
			m_memoryCounter +=
					(int) (imgDescriptors.rows * cols * sizeof(uchar));
		} else {
			m_memoryCounter +=
					(int) (imgDescriptors.rows * cols * sizeof(float));
		}
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