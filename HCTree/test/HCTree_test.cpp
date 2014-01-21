/*
 * HCTree_test.cpp
 *
 *  Created on: Jan 20, 2014
 *      Author: andresf
 */

#include <gtest/gtest.h>
#include <HCTree.hpp>
#include <FileUtils.hpp>

TEST(HCTree, InstantiateOnHeap) {
	cv::Ptr<vlr::HCTree> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new vlr::HCTree();

	EXPECT_TRUE(tree != NULL);
}

TEST(HCTree, InstantiateOnStack) {

	vlr::HCTree tree;

	EXPECT_TRUE(tree.empty());

}

TEST(HCTree, InitParams) {

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");
	DynamicMat data(keysFilenames);
	vlr::HCTree tree(data);

	cv::Mat descriptors;
	FileUtils::loadDescriptors(keysFilenames[0], descriptors);

	EXPECT_TRUE(int(tree.getVeclen()) == descriptors.cols);

}

TEST(HCTree, TreeBuilding) {

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");
	DynamicMat data(keysFilenames);

	vlr::HCTree tree(data);

	EXPECT_TRUE(tree.empty());

	tree.build();

	EXPECT_FALSE(tree.empty());

}

TEST(HCTree, SaveLoad) {

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");
	DynamicMat data(keysFilenames);

	vlr::HCTree tree(data);

	tree.build();

	tree.save("test_tree.yaml.gz");

	vlr::HCTree treeCopy;
	treeCopy.load("test_tree.yaml.gz");

	EXPECT_TRUE(tree == treeCopy);

}
