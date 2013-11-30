/*
 * InvertedIndex.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <cstring>

#include <InvertedIndex.h>

InvertedIndex::InvertedIndex() {
}

InvertedIndex::~InvertedIndex() {
}

bool InvertedIndex::operator==(const InvertedIndex &other) const {
	if (this->size() != other.size()) {
		return false;
	}
	for (size_t i = 0; i < this->size(); i++) {
		//		int idWordA = this->at(i)->word_id;
		//		int idWordB = other.at(i)->word_id;
		//		ASSERT_TRUE(idWordA == idWordB);
		//		double weightWordA = this->at(i)->weight;
		//		double weightWordB = other.at(i)->weight;
		//		ASSERT_TRUE(weightWordA == weightWordB);
		//
		//		size_t invertedFileLengthWordA = this->at(i)->image_list.size();
		//		size_t invertedFileLengthWordB = other.at(i)->image_list.size();
		//		ASSERT_TRUE(invertedFileLengthWordA == invertedFileLengthWordB);
	}
	return true;
}
