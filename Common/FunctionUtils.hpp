/*
 * FunctionUtils.hpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#ifndef FUNCTIONUTILS_HPP_
#define FUNCTIONUTILS_HPP_

#include <vector>
#include <queue>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <sstream>

class HtmlResultsWriter {
public:

	static HtmlResultsWriter& getInstance() {
		static HtmlResultsWriter instance; // Guaranteed to be destroyed.
		// Instantiated on first use.
		return instance;
	}

	void writeHeader(FILE *f, int num_nns);

	void writeRow(FILE *f, const std::string &query, cv::Mat& scores,
			cv::Mat& perm, int num_nns,
			const std::vector<std::string> &db_images);

	void writeFooter(FILE *f);

	std::string getHtml() const;

private:
	// Make the constructor private so that it cannot be instantiated from outside
	HtmlResultsWriter() {
	}
	;

	// Make private the copy constructor and the assignment operator
	// to prevent obtaining copies of the singleton
	HtmlResultsWriter(HtmlResultsWriter const&); // Don't Implement
	void operator=(HtmlResultsWriter const&); // Don't implement

	void basifyFilename(const char *filename, char *base);

protected:
	std::stringstream html;
};

struct image {
	int imgIdx; // Index of the image it represents
	int startIdx; // (inclusive) Starting index in the merged descriptors matrix
	image() :
			imgIdx(-1), startIdx(-1) {
	}
};

static std::vector<image> DEFAULT_INDICES;
static std::vector<std::string> DEFAULT_KEYS;

class DynamicMat {

public:

	DynamicMat(std::vector<image>& descriptorsIndices = DEFAULT_INDICES,
			std::vector<std::string>& keysFilenames = DEFAULT_KEYS,
			int descriptorCount = 0, int descriptorLength = 0,
			int descriptorType = -1);

	// Copy constructor
	DynamicMat(const DynamicMat& other);

	// Assignment operators
	DynamicMat& operator =(const DynamicMat& other);

	virtual ~DynamicMat();

	const std::vector<image>& getDescriptorsIndices() const {
		return m_descriptorsIndices;
	}

	const std::vector<std::string>& getKeysFilenames() const {
		return m_keysFilenames;
	}

	cv::Mat row(int descriptorIndex);

	int type() const;

	bool empty() const;

protected:

	std::vector<image> m_descriptorsIndices;
	std::vector<std::string> m_keysFilenames;
	std::map<int, cv::Mat> descBuffer;
	std::queue<int> addingOrder;

public:
	static const int MAX_MEM = 367000000; // ~350 MBytes
	int rows = 0;
	int cols = 0;
	int m_memoryCounter = 0;

protected:
	int m_descriptorType = -1;

};

namespace FunctionUtils {

void printKeypoints(std::vector<cv::KeyPoint>& keypoints);

void printDescriptors(const cv::Mat& descriptors);

/**
 * Function for printing to standard output the parameters of <i>algorithm</i>
 *
 * @param algorithm A pointer to an object of type cv::Algorithm
 */
void printParams(cv::Ptr<cv::Algorithm> algorithm);

/**
 * Counts the number of bits equal to 1 in a specified number.
 *
 * @param i Number to count bits on
 * @return Number of bits equal to 1
 */
int NumberOfSetBits(int i);

int BinToDec(const cv::Mat& binRow);

void split(const std::string &s, char delim, std::vector<std::string> &tokens);

} // namespace FunctionUtils

#endif /* FUNCTIONUTILS_HPP_ */
