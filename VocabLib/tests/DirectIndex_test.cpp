/*
 * DirectIndex_test.cpp
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>

#include <DirectIndex.hpp>

TEST(DirectIndex, InstantiateOnHeap) {

	cv::Ptr<bfeat::DirectIndex> di;

	EXPECT_TRUE(di == NULL);

	di = new bfeat::DirectIndex();

	EXPECT_TRUE(di != NULL);
}

TEST(DirectIndex, AddFeature) {

	cv::Ptr<bfeat::DirectIndex> di = new bfeat::DirectIndex(3);

	EXPECT_TRUE(di->getLevel() == 3);

	size_t nodes[5] = { 3, 6, 7, 14 };

	for (size_t imgIdx = 0; imgIdx < 5; ++imgIdx) {
		for (size_t nodeIdx = 0; nodeIdx < 4; ++nodeIdx) {
			for (size_t featIdx = 0; featIdx < 3000; ++featIdx) {
				di->addFeature(imgIdx, nodes[nodeIdx], featIdx);
			}
		}
	}

	// Check number of images in the index
	EXPECT_TRUE(di->size() == 5);

	for (size_t imgIdx = 0; imgIdx < 5; ++imgIdx) {
		bfeat::TreeNode node = di->lookUpImg(imgIdx);

		// Check each image index has four nodes
		EXPECT_TRUE(node.size() == 4);

		for (bfeat::TreeNode::iterator it = node.begin(); it != node.end();
				it++) {
			// Check each nodes has 3000 features
			EXPECT_TRUE(it->second.size() == 3000);
		}
	}
}

TEST(DirectIndex, SaveLoad) {

	cv::Ptr<bfeat::DirectIndex> indexOne = new bfeat::DirectIndex();

	size_t nodes[5] = { 3, 6, 7, 14 };

	for (size_t imgIdx = 0; imgIdx < 5; ++imgIdx) {
		for (size_t nodeIdx = 0; nodeIdx < 4; ++nodeIdx) {
			for (size_t featIdx = 0; featIdx < 10; ++featIdx) {
				indexOne->addFeature(imgIdx, nodes[nodeIdx], featIdx);
			}
		}
	}

	indexOne->save("test_di.yaml.gz");
	cv::Ptr<bfeat::DirectIndex> indexTwo = new bfeat::DirectIndex();
	indexTwo->load("test_di.yaml.gz");

	// Check number of images in the index
	EXPECT_TRUE(indexOne->size() == indexTwo->size());

	for (size_t imgIdx = 0; imgIdx < indexOne->size(); ++imgIdx) {
		bfeat::TreeNode nodeOne = indexOne->lookUpImg(imgIdx), nodeTwo =
				indexTwo->lookUpImg(imgIdx);

		// Check each image index has four nodes
		EXPECT_TRUE(nodeOne.size() == nodeTwo.size());

		bfeat::TreeNode::iterator itOne = nodeOne.begin(), itTwo =
				nodeTwo.begin();

		for (; itOne != nodeOne.end() && itTwo != nodeTwo.end();
				++itOne, ++itTwo) {
			// Check each nodes has 3000 features
			EXPECT_TRUE(itOne->second.size() == itTwo->second.size());
		}
	}

}
