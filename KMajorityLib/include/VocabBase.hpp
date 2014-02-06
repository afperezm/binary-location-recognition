/*
 * VocabBase.hpp
 *
 *  Created on: Feb 3, 2014
 *      Author: andresf
 */

#ifndef VOCABBASE_HPP_
#define VOCABBASE_HPP_

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

};

} /* namespace vlr */

#endif /* VOCABBASE_HPP_ */
