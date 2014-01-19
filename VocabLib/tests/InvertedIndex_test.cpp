/*
 * InvertedIndex_test.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>

#include <InvertedIndex.hpp>

TEST(ImageCount, EmptyConstructor) {

	vlr::ImageCount counter;

	EXPECT_TRUE(counter.m_count == 0);
	EXPECT_TRUE(counter.m_index == 0.0);

}

TEST(ImageCount, DefaultConstructor) {

	vlr::ImageCount counter(10, 30.0);

	EXPECT_TRUE(counter.m_index == 10);
	EXPECT_TRUE(counter.m_count == 30.0);

}

TEST(Word, EmptyConstructor) {

	vlr::Word w;

	EXPECT_TRUE(w.m_weight == 0.0);

}

TEST(Word, DefaultConstructor) {

	vlr::Word w(180.0);

	EXPECT_TRUE(w.m_weight == 180.0);

}

TEST(InvertedIndex, InstantiateOnHeap) {

	cv::Ptr<vlr::InvertedIndex> index;

	EXPECT_TRUE(index == NULL);

	index = new vlr::InvertedIndex();

	EXPECT_TRUE(index != NULL);

}

TEST(InvertedIndex, InstantiateOnStack) {

	vlr::InvertedIndex index;

	EXPECT_TRUE(index.size() == 0);

}

TEST(InvertedIndex, SaveLoad) {

	vlr::InvertedIndex index;

	vlr::Word w(1.0);
	for (int imgIdx = 0; imgIdx < 5; ++imgIdx) {
		w.m_imageList.clear();
		w.m_imageList.push_back(vlr::ImageCount(imgIdx, 1.0));
		index.push_back(w);
	}

	index.save("test_inv_idx.yaml.gz");

	vlr::InvertedIndex indexLoaded;
	indexLoaded.load("test_inv_idx.yaml.gz");

	EXPECT_TRUE(index == indexLoaded);

}
