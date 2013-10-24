/*
 * bin_hierarchical_clustering_index_test.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <VocabTree.h>
#include <opencv2/core/core.hpp>

TEST(VocabTree, Instantiation) {
	cv::Ptr<cvflann::VocabTreeBase> tree;
	tree = new cvflann::VocabTree<float, cv::L2<float> >();

	std::cout << "a is of type: " << typeid(tree).name() << std::endl;

	ASSERT_NE(NULL, tree);
}
