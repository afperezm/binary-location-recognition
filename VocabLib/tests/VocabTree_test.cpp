/*
 * bin_hierarchical_clustering_index_test.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <VocabTree.h>
#include <opencv2/core/core.hpp>
#include <limits.h>

TEST(VocabTree, Instantiation) {
	cv::Ptr<cvflann::VocabTreeBase> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new cvflann::VocabTree<float, cv::L2<float> >();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTree, LoadSaveReal) {

	// 128 dimension in SIFT
	cv::Mat data(3000, 128, cv::DataType<float>::type);

	cv::randn(data, cv::Scalar(128), cv::Scalar(128));

	cv::Ptr<cvflann::VocabTree<float, cv::L2<float> > > tree =
			new cvflann::VocabTree<float, cv::L2<float> >(data);

	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<cvflann::VocabTree<float, cv::L2<float> > > treeLoad =
			new cvflann::VocabTree<float, cv::L2<float> >();

	treeLoad->load("test_tree.yaml.gz");

	ASSERT_TRUE(tree->size() == treeLoad->size());

	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

}

TEST(VocabTree, LoadSaveBinary) {

	// Number of columns is the number of bytes
	cv::Mat data(3000, 32 / 8, cv::DataType<uchar>::type);

	cv::randn(data, cv::Scalar(floor(UCHAR_MAX / 2)),
			cv::Scalar(floor(UCHAR_MAX / 2)));

	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > tree;
	tree = new cvflann::VocabTree<uchar, cv::Hamming>(data);

	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > treeLoad =
			new cvflann::VocabTree<uchar, cv::Hamming>();

	treeLoad->load("test_tree.yaml.gz");

	ASSERT_TRUE(tree->size() == treeLoad->size());

	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

}
