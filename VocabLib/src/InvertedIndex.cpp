/*
 * InvertedIndex.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <opencv2/core/core.hpp>

#include <InvertedIndex.hpp>

#include <assert.h>
#include <iostream>
#include <fstream>

namespace vlr {

InvertedIndex::InvertedIndex() :
		m_numDbImages(0) {
}

// --------------------------------------------------------------------------

InvertedIndex::~InvertedIndex() {
}

// --------------------------------------------------------------------------

bool InvertedIndex::operator==(const InvertedIndex &other) const {
	// Check indices have same size
	if (size() != other.size()) {
		printf("Indices have different size, [%lu] against [%lu]\n", size(),
				other.size());
		return false;
	}
	// Check words are equal
	for (int i = 0; i < int(size()); ++i) {
		if (at(i) != other.at(i)) {
			printf("Words at position [%d] are unequal\n", i);
			return false;
		}
	}
	return true;
}

// --------------------------------------------------------------------------

void InvertedIndex::save(const std::string& filename) const {

	if (empty() == true) {
		throw std::runtime_error("[VocabTree::save] "
				"Vocabulary is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[VocabTree::saveInvertedIndex] "
				"Unable to open file "
				"[" + filename + "] for writing");
	}

	fs << "NumDBImages" << int(m_numDbImages);

	fs << "Words" << "[";

	for (size_t i = 0; i < size(); ++i) {
		fs << "{";

		fs << "weight" << at(i).m_weight;
		fs << "imageList" << "[";
		for (ImageCount img : at(i).m_imageList) {
			fs << "{:" << "m_index" << int(img.m_index) << "m_count"
					<< img.m_count << "}";
		}
		fs << "]";

		fs << "}";
	}

	fs << "]";

	fs.release();
}

// --------------------------------------------------------------------------

void InvertedIndex::load(const std::string& filename) {

	// Clear index
	clear();

	// Initializing variables
	std::ifstream inputZippedFileStream;
	boost::iostreams::filtering_istream inputFileStream;
	std::string line, wordHeader = "   -", weightHeader = "weight", pairHeader =
			"{";
	int index, wordId = -1;
	float count;

	// Open file
	inputZippedFileStream.open(filename.c_str(),
			std::fstream::in | std::fstream::binary);

	// Check file
	if (inputZippedFileStream.good() == false) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Unable to open file [" + filename + "] for reading");
	}

	try {
		inputFileStream.push(boost::iostreams::gzip_decompressor());
		inputFileStream.push(inputZippedFileStream);

		// Load/Skip first line
		getline(inputFileStream, line);

		// Load number of images
		getline(inputFileStream, line);
		sscanf(line.c_str(), "%*s %d", &m_numDbImages);

		// Load/Skip words vector header
		getline(inputFileStream, line);

		// Load list from file
		while (getline(inputFileStream, line)) {
			if (line.compare(wordHeader) == 0) {
				++wordId;
				push_back(vlr::Word());
			} else if (line.find(weightHeader) != std::string::npos) {
				sscanf(line.c_str(), "%*s %lf", &at(wordId).m_weight);
			} else if (line.find(pairHeader) != std::string::npos) {
				line.replace(line.find('-'), 1, " ");
				std::replace(line.begin(), line.end(), ':', ' ');
				std::replace(line.begin(), line.end(), ',', ' ');
				std::replace(line.begin(), line.end(), '{', ' ');
				std::replace(line.begin(), line.end(), '}', ' ');
				sscanf(line.c_str(), "%*s %d %*s %f", &index, &count);
				ImageCount img(index, count);
				at(wordId).m_imageList.push_back(img);
			}
		}
	} catch (const boost::iostreams::gzip_error& e) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Got error while parsing file [" + std::string(e.what()) + "]");
	}

	// Close file
	inputZippedFileStream.close();

}

} /* namespace vlr */
