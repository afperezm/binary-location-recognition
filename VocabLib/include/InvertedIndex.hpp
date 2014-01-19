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

	ImageCount() :
			m_index(0), m_count(0.0) {
	}

	ImageCount(unsigned int index, float count) :
			m_index(index), m_count(count) {
	}

	// Index of the database image this entry corresponds to
	unsigned int m_index;

	// (Weighted, normalized) Count of how many times this feature appears
	float m_count;

};

class Word {

public:

	Word() :
			m_id(-1), m_weight(0.0) {
	}

	Word(int id, double weight) :
			m_id(id), m_weight(weight) {
	}

	// Word id (only for terminal nodes)
	int m_id;

	// Weight (only for terminal nodes)
	double m_weight;

	// Inverse document/image list (only for terminal nodes)
	std::vector<ImageCount> m_imageList;

};

class InvertedIndex: public std::vector<Word> {

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
	void saveInvertedIndex(const std::string& filename) const;

	/**
	 * Loads the inverted index from a file stream.
	 *
	 * @param filename - The name of the file stream from where to load the index
	 */
	void loadInvertedIndex(const std::string& filename);

protected:

	// Number of DB images, used for creating the scores matrix
	uint m_numDbImages;

};

} /* namespace vlr */

#endif /* INVERTEDINDEX_H_ */
