/*
 * HtmlResultsWriter.hpp
 *
 *  Created on: Nov 10, 2013
 *      Author: andresf
 */

#ifndef HTMLRESULTSWRITER_HPP_
#define HTMLRESULTSWRITER_HPP_

#include <sstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

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

#endif /* HTMLRESULTSWRITER_HPP_ */
