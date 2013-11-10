/*
 * FunctionUtils.hpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#ifndef FUNCTIONUTILS_HPP_
#define FUNCTIONUTILS_HPP_

#include <vector>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

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
