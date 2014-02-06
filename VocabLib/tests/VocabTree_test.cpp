/*
 * VocabTree_test.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#include <limits.h>

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabTree.h>
#include <VocabDB.hpp>

TEST(VocabTreeReal, InstantiateOnHeap) {
	cv::Ptr<vlr::VocabTreeReal> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new vlr::VocabTreeReal();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTreeBinary, InstantiateOnHeap) {
	cv::Ptr<vlr::VocabTreeBin> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new vlr::VocabTreeBin();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTreeReal, LoadSave) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_1.yaml.gz");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<vlr::VocabTreeReal> tree = new vlr::VocabTreeReal(data);

	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<vlr::VocabTreeReal> treeLoad = new vlr::VocabTreeReal();

	treeLoad->load("test_tree.yaml.gz");

	// Check tree structure is the same
	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

	// Check vocabulary size is the same
	ASSERT_TRUE(tree->size() == treeLoad->size());

}

TEST(VocabTreeBinary, LoadSave) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");

	vlr::Mat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<vlr::VocabTreeBin> tree;
	tree = new vlr::VocabTreeBin(data);

	cvflann::seed_random(unsigned(std::time(0)));
	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<vlr::VocabTreeBin> treeLoad = new vlr::VocabTreeBin();

	treeLoad->load("test_tree.yaml.gz");

	// Check tree structure is the same
	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

	// Check vocabulary size is the same
	ASSERT_TRUE(tree->size() == treeLoad->size());

}
