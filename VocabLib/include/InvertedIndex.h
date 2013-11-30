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
public:
	InvertedIndex();
	virtual ~InvertedIndex();

	bool operator==(const InvertedIndex &other) const;

protected:

};

#endif /* INVERTEDINDEX_H_ */
