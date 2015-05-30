/*
 * DynamicMat.cpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#include <DynamicMat.hpp>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <libmemcached-1.0/memcached.hpp>
#include <opencv2/core/mat.hpp>
#include <cstring>
#include <stdexcept>
#include <unistd.h>

namespace vlr {

Mat::Mat() :
		m_descriptorType(-1), m_elemSize(0), client("--SERVER=127.0.0.1:21201"), rows(
				0), cols(0) {
#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing empty\n");
#endif

}

// --------------------------------------------------------------------------

Mat::Mat(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by copy\n");
#endif

	m_descriptorType = other.type();
	m_elemSize = other.elemSize();
	client = other.client;
	rows = other.rows;
	cols = other.cols;

}

// --------------------------------------------------------------------------

Mat::Mat(std::vector<std::string>& descriptorsFilenames) :
		client("--SERVER=127.0.0.1:21201") {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing using filenames\n");
#endif

	cv::Mat descriptors;

	int descCount = 0, descLen = 0, descType = -1, imgIdx = 0;
	size_t descElemSize = 0;

	double mytime = cv::getTickCount();

	// Initialize descriptors index
	for (std::string& descriptorsFilename : descriptorsFilenames) {

		printf("[DynamicMat] Loading descriptors file [%04d/%04lu]\n",
				imgIdx + 1, descriptorsFilenames.size());

		// Load descriptors
		FileUtils::loadDescriptors(descriptorsFilename, descriptors);

		// Increase counter if descriptors matrix is not empty
		if (descriptors.empty() == false) {
			// If initialized check descriptors length
			if (descLen != 0) {
				// Recall that all descriptors must be of the same length
				CV_Assert(descLen == descriptors.cols);
			} else {
				descLen = descriptors.cols;
			}
			// If initialized check descriptors type
			if (descType != -1) {
				// Recall that all descriptors must be of the same type
				CV_Assert(descType == descriptors.type());
			} else {
				descType = descriptors.type();
			}
			// If initialized check descriptors element size
			if (descElemSize != 0) {
				// Recall that all descriptors must have the same element size
				CV_Assert(descElemSize == descriptors.elemSize());
			} else {
				descElemSize = descriptors.elemSize();
			}
			// Adding descriptors to the persistent cache
			for (int i = 0; i < descriptors.rows; ++i) {
				cv::Mat row = descriptors.row(i);
				uchar* p_row = row.data;
				std::vector<char> value(p_row, p_row + row.step * row.rows);
				std::stringstream ss;
				ss << descCount + i;
				std::string key = ss.str();
#if DYNMATVERBOSE
				printf("[DynamicMat] Adding descriptor [%s]\n", key.c_str());
#endif
				bool valueAdded = client.set(key, value, 0, 0);
				if (valueAdded == false) {
					throw std::runtime_error("Unable to add descriptor to cache");
				}
			}
			// Increase descriptors counter
			descCount += descriptors.rows;
		}

		// Increase images counter
		++imgIdx;
	}

	mytime = (double(cv::getTickCount()) - mytime) / cv::getTickFrequency() * 1000;

#if DYNMATVERBOSE
	printf("[DynamicMat] Initialized descriptors index in [%lf] ms\n", mytime);
#endif

	rows = descCount;
	cols = descLen;
	m_descriptorType = descType;
	m_elemSize = descElemSize;

	CV_Assert(cols > 0 && descElemSize > 0);

	client.flush(0);

}

// --------------------------------------------------------------------------

Mat::~Mat() {
#if DYNMATVERBOSE
	printf("[DynamicMat] Destroying\n");
#endif
}

// --------------------------------------------------------------------------

Mat& Mat::operator=(const Mat& other) {

#if DYNMATVERBOSE
	printf("[DynamicMat] Initializing by assignment\n");
#endif

	m_descriptorType = other.type();
	m_elemSize = other.elemSize();
	client = other.client;
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

	std::stringstream ss;
	ss << descriptorIdx;
	std::vector<char> value;
	client.get(ss.str(), value);

	cv::Mat descriptor(1, cols, m_descriptorType);
	memcpy(reinterpret_cast<char*>(descriptor.data), reinterpret_cast<char*>(value.data()), value.size());

	return descriptor;
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

} /* namespace vlr */
