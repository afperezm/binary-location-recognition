/*
 * InvertedIndex.h
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#ifndef INVERTEDINDEX_H_
#define INVERTEDINDEX_H_

#include <vector>

class InvertedIndex: public std::vector<int> {

protected:

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

};

#endif /* INVERTEDINDEX_H_ */
