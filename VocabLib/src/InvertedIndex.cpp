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
	if (this->size() != other.size()) {
		return false;
	}
	for (size_t i = 0; i < this->size(); i++) {
		int idWordA = this->at(i).m_id;
		int idWordB = other.at(i).m_id;
		assert(idWordA == idWordB);
		double weightWordA = this->at(i).m_weight;
		double weightWordB = other.at(i).m_weight;
		assert(weightWordA == weightWordB);

		size_t invertedFileLengthWordA = this->at(i).m_imageList.size();
		size_t invertedFileLengthWordB = other.at(i).m_imageList.size();
		assert(invertedFileLengthWordA == invertedFileLengthWordB);
	}
	return true;
}

// --------------------------------------------------------------------------

void InvertedIndex::saveInvertedIndex(const std::string& filename) const {

	if (this->empty() == true) {
		throw std::runtime_error("[VocabTree::saveInvertedIndex] "
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

	for (size_t i = 0; i < this->size(); ++i) {
		fs << "{";

		fs << "weight" << this->at(i).m_weight;
		fs << "imageList" << "[";
		for (ImageCount img : this->at(i).m_imageList) {
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

void InvertedIndex::loadInvertedIndex(const std::string& filename) {

	if (this->empty() == true) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Vocabulary is empty");
	}

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
			} else if (line.find(weightHeader) != std::string::npos) {
				sscanf(line.c_str(), "%*s %lf", &this->at(wordId).m_weight);
			} else if (line.find(pairHeader) != std::string::npos) {
				line.replace(line.find('-'), 1, " ");
				std::replace(line.begin(), line.end(), ':', ' ');
				std::replace(line.begin(), line.end(), ',', ' ');
				std::replace(line.begin(), line.end(), '{', ' ');
				std::replace(line.begin(), line.end(), '}', ' ');
				sscanf(line.c_str(), "%*s %d %*s %f", &index, &count);
				ImageCount img(index, count);
				this->at(wordId).m_imageList.push_back(img);
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
