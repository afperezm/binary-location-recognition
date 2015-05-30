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

#include <libmemcached/memcached.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#define HOST 127.0.0.1
#define PORT 21201

namespace vlr {

class Mat {

public:

private:

	int m_descriptorType = -1;
	size_t m_elemSize = 0;

	/** Attributes of the cache **/
	memcache::Memcache client = NULL;

public:

	int rows = 0;
	int cols = 0;

public:

	/**
	 * Class empty constructor.
	 */
	Mat();

	/**
	 * Copy constructor.
	 *
	 * @param other - Reference to an instance where to copy properties from
	 */
	Mat(const Mat& other);

	/**
	 * Class constructor.
	 *
	 * @param keysFilenames - Reference to a vector of descriptor filenames
	 */
	Mat(std::vector<std::string>& keysFilenames);

	/**
	 * Class destroyer.
	 */
	virtual ~Mat();

	/**
	 * Assignment operator.
	 *
	 * @param other - Reference to an instance where to copy properties from
	 * @return reference to the new instance
	 */
	Mat& operator=(const Mat& other);

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

	size_t elemSize() const;

	/**
	 * Returns true if the matrix is empty.
	 *
	 * @return true if rows counter is zero, false otherwise
	 */
	bool empty() const;

};

static vlr::Mat DEFAULT_INPUTDATA = vlr::Mat();

} /* namespace vlr */

#endif /* DYNAMICMAT_HPP_ */
