/*
 * InvertedIndex.hpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#ifndef INVERTEDINDEX_H_
#define INVERTEDINDEX_H_

#include <vector>

namespace vlr {

class ImageCount {

public:

	// Index of the database image this entry corresponds to
	unsigned int m_index;

	// (Weighted, normalized) Count of how many times this feature appears
	float m_count;

public:

	/**
	 * Empty constructor.
	 */
	ImageCount() :
			m_index(0), m_count(0.0) {
	}

	/**
	 * Class constructor.
	 *
	 * @param index - Image index
	 * @param count - Image count
	 */
	ImageCount(unsigned int index, float count) :
			m_index(index), m_count(count) {
	}

	/**
	 * Equality operator.
	 *
	 * @param other
	 * @return true if objects are equal, false otherwise
	 */
	bool operator==(const ImageCount &other) const {

		if (m_index != other.m_index) {
			fprintf(stderr, "Image count objects refer to different images\n");
			return false;
		}

		if (m_count != other.m_count) {
			fprintf(stderr, "Image count objects have different count\n");
			return false;
		}

		return true;
	}

	/**
	 * Inequality operator.
	 *
	 * @param other
	 * @return true if objects are unequal, false otherwise
	 */
	bool operator!=(const ImageCount &other) const {
		return !operator==(other);
	}

};

class Word {

public:

	// Weight (only for terminal nodes)
	double m_weight;

	// Inverse document/image list (only for terminal nodes)
	std::vector<ImageCount> m_imageList;

public:

	/**
	 * Empty constructor.
	 */
	Word() :
			m_weight(0.0) {
	}

	/**
	 * Class constructor.
	 *
	 * @param weight - Word weight
	 */
	Word(double weight) :
			m_weight(weight) {
	}

	/**
	 * Equality operator.
	 *
	 * @param other
	 * @return true if objects are equal, false otherwise
	 */
	bool operator==(const Word &other) const {

		// Check weights
		if (m_weight != other.m_weight) {
			fprintf(stderr, "Words have different weights\n");
			return false;
		}

		// Check inverted files have same length
		if (m_imageList.size() != other.m_imageList.size()) {
			fprintf(stderr, "Words inverted files have different length\n");
			return false;
		}

		// Check inverted files are equal
		for (int j = 0; j < int(m_imageList.size()); ++j) {
			if (m_imageList.at(j) != other.m_imageList.at(j)) {
				fprintf(stderr, "Word inverted files are not equal\n");
				return false;
			}
		}

		return true;
	}

	/**
	 * Inequality operator.
	 *
	 * @param other
	 * @return true if objects are unequal, false otherwise
	 */
	bool operator!=(const Word &other) const {
		return !operator==(other);
	}

};

class InvertedIndex: public std::vector<Word> {

public:

	// Number of database images
	uint m_numDbImages;

public:

	/**
	 * Class constructor.
	 */
	InvertedIndex();

	/**
	 * Class destroyer.
	 */
	virtual ~InvertedIndex();

	/**
	 * Equality operator.
	 *
	 * @param other
	 * @return true if indexes are equal, false otherwise
	 */
	bool operator==(const InvertedIndex &other) const;

	/**
	 * Saves the inverted index to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the index
	 */
	void save(const std::string& filename) const;

	/**
	 * Loads the inverted index from a file stream.
	 *
	 * @param filename - The name of the file stream from where to load the index
	 */
	void load(const std::string& filename);

};

} /* namespace vlr */

#endif /* INVERTEDINDEX_H_ */
