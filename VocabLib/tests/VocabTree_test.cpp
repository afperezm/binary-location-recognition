/*
 * VocabTree_test.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#include <ctime>
#include <limits.h>

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabTree.h>

TEST(VocabTree, Instantiation) {
	cv::Ptr<bfeat::VocabTreeBase> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new bfeat::VocabTree<float, cv::L2<float> >();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTree, LoadSaveReal) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_1.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<bfeat::VocabTree<float, cv::L2<float> > > tree =
			new bfeat::VocabTree<float, cv::L2<float> >(data);

	cvflann::seed_random(unsigned(std::time(0)));
	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<bfeat::VocabTree<float, cv::L2<float> > > treeLoad =
			new bfeat::VocabTree<float, cv::L2<float> >();

	treeLoad->load("test_tree.yaml.gz");

	// Check tree structure is the same
	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

	// Check vocabulary size is the same
	ASSERT_TRUE(tree->size() == treeLoad->size());

}

TEST(VocabTree, LoadSaveBinary) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<bfeat::VocabTree<uchar, cv::Hamming> > tree;
	tree = new bfeat::VocabTree<uchar, cv::Hamming>(data);

	cvflann::seed_random(unsigned(std::time(0)));
	tree->build();

	tree->save("test_tree.yaml.gz");

	cv::Ptr<bfeat::VocabTree<uchar, cv::Hamming> > treeLoad =
			new bfeat::VocabTree<uchar, cv::Hamming>();

	treeLoad->load("test_tree.yaml.gz");

	// Check tree structure is the same
	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

	// Check vocabulary size is the same
	ASSERT_TRUE(tree->size() == treeLoad->size());

}

TEST(VocabTree, TestDatabase) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_1.yaml.gz");

	DynamicMat data(keysFilenames);

	cv::Ptr<bfeat::VocabTreeBase> tree = new bfeat::VocabTree<float,
			cv::L2<float> >(data);

	tree->build();
	tree->save("test_tree.yaml.gz");
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<bfeat::VocabTreeBase> db = new bfeat::VocabTree<float,
			cv::L2<float> >();

	db->load("test_tree.yaml.gz");

	db->clearDatabase();

	bool gotException = false;
	uint i = 0;
	for (std::string keyFileName : keysFilenames) {
		try {
			FileUtils::loadDescriptors(keyFileName, imgDescriptors);
			db->addImageToDatabase(i, imgDescriptors);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			gotException = true;
		}
		i++;
	}

	// Assert all images where inserted without any problem
	ASSERT_FALSE(gotException);

	cv::Mat dbBowVector, sumResult;

	// Asserting inverted files are not empty anymore
	for (size_t imgIdx = 0; imgIdx < keysFilenames.size(); imgIdx++) {
		db->getDbBoWVector(imgIdx, dbBowVector);
		cv::reduce(dbBowVector, sumResult, 1, CV_REDUCE_SUM);
		ASSERT_TRUE(sumResult.rows == 1);
		ASSERT_TRUE(sumResult.cols == 1);
		ASSERT_TRUE(sumResult.at<float>(0, 0) != 0);
	}

	db->computeWordsWeights(bfeat::TF_IDF);

	db->createDatabase();

	db->normalizeDatabase(cv::NORM_L1);

	// Asserting DB BoW vectors values are in the range [0,1]
	for (size_t imgIdx = 0; imgIdx < keysFilenames.size(); imgIdx++) {
		db->getDbBoWVector(imgIdx, dbBowVector);
		ASSERT_TRUE(dbBowVector.rows == 1);
		for (int i = 0; i < dbBowVector.cols; i++) {
			ASSERT_TRUE(dbBowVector.at<float>(0, i) >= 0);
			ASSERT_TRUE(dbBowVector.at<float>(0, i) <= 1);
		}
	}

	cv::Ptr<bfeat::VocabTreeBase> dbLoad = new bfeat::VocabTree<float,
			cv::L2<float> >();

	dbLoad->load("test_tree.yaml.gz");

	db->saveInvertedIndex("test_idf.yaml.gz");
	dbLoad->loadInvertedIndex("test_idf.yaml.gz");

	// TODO Test inverted indices are equal

	// Querying the tree using the same documents used for building it,
	// the top result must be the document itself and hence the score must be 1
	i = 0;
	for (std::string keyFileName : keysFilenames) {
		cv::Mat scores;

		imgDescriptors = cv::Mat();
		FileUtils::loadDescriptors(keyFileName, imgDescriptors);
		db->scoreQuery(imgDescriptors, scores, cv::NORM_L1);
		// Check that scores has the right type
		EXPECT_TRUE(cv::DataType<float>::type == scores.type());
		// Check that scores is a row vector
		EXPECT_TRUE(1 == scores.rows);
		// Check all DB images have been scored
		EXPECT_TRUE((int )keysFilenames.size() == scores.cols);

		cv::Mat perm;
		cv::sortIdx(scores, perm, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
		EXPECT_TRUE(scores.rows == perm.rows);
		EXPECT_TRUE(scores.cols == perm.cols);

		EXPECT_TRUE((int )i == perm.at<int>(0, 0));
		EXPECT_TRUE(round(scores.at<float>(0, perm.at<int>(0, 0))) == 1.0);
		i++;
	}

}
