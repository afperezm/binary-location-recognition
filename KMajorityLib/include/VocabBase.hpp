/*
 * VocabBase.hpp
 *
 *  Created on: Feb 3, 2014
 *      Author: andresf
 */

#ifndef VOCABBASE_HPP_
#define VOCABBASE_HPP_

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

namespace vlr {

class VocabBase {

public:

	/**
	 * Virtual destroyer to enable destruction from a subclass.
	 */
	virtual ~VocabBase() {
	}

	/**
	 * Builds the vocabulary.
	 */
	virtual void build() = 0;

	/**
	 * Saves the vocabulary to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the vocabulary
	 */
	virtual void save(const std::string& filename) const = 0;

	/**
	 * Returns the vocabulary size.
	 *
	 * @return the number of words
	 */
	virtual size_t size() const = 0;

	static std::string loadVocabType(const std::string& filename) {

		std::ifstream inputZippedFileStream;
		boost::iostreams::filtering_istream inputFileStream;

		std::string line, field;
		std::stringstream ss;

		enum vocabFields {
			type
		};
		std::string treeFieldsNames[] = { "type:" };

		// Open file
		inputZippedFileStream.open(filename.c_str(),
				std::fstream::in | std::fstream::binary);

		// Check file
		if (inputZippedFileStream.good() == false) {
			throw std::runtime_error("[VocabBase::loadVocabType] "
					"Unable to open file [" + filename + "] for reading");
		}

		std::string vocabType = "UNKNOWN";

		try {
			inputFileStream.push(boost::iostreams::gzip_decompressor());
			inputFileStream.push(inputZippedFileStream);

			while (getline(inputFileStream, line)) {
				ss.clear();
				ss.str(line);
				ss >> field;
				if (field.compare(treeFieldsNames[type]) == 0) {
					ss >> vocabType;
					break;
				}
			}

		} catch (const boost::iostreams::gzip_error& e) {
			throw std::runtime_error(
					"[VocabTree::load] "
							"Got error while parsing file ["
							+ std::string(e.what()) + "]");
		}

		// Close file
		inputZippedFileStream.close();

		return vocabType;
	}

};

} /* namespace vlr */

#endif /* VOCABBASE_HPP_ */
