/*
 * bin_hierarchical_clustering_index_test.cpp
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 */

#include <ctime>
#include <limits.h>

#include <gtest/gtest.h>
#include <opencv2/core/core.hpp>

#include <FileUtils.hpp>
#include <VocabTree.h>

TEST(VocabTree, Instantiation) {
	cv::Ptr<cvflann::VocabTreeBase> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new cvflann::VocabTree<float, cv::L2<float> >();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTree, LoadSaveReal) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_1.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<cvflann::VocabTree<float, cv::L2<float> > > tree =
			new cvflann::VocabTree<float, cv::L2<float> >(data);

	cvflann::seed_random(unsigned(std::time(0)));
	tree->build();

	tree->save("test_tree.yaml.gz");
	tree->saveInvertedIndex("test_idf.yaml.gz");

	cv::Ptr<cvflann::VocabTree<float, cv::L2<float> > > treeLoad =
			new cvflann::VocabTree<float, cv::L2<float> >();

	treeLoad->load("test_tree.yaml.gz");
	treeLoad->loadInvertedIndex("test_idf.yaml.gz");

	ASSERT_TRUE(tree->size() == treeLoad->size());

	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

}

TEST(VocabTree, LoadSaveBinary) {

	/////////////////////////////////////////////////////////////////////
	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("brief_0.yaml.gz");
	keysFilenames.push_back("brief_1.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > tree;
	tree = new cvflann::VocabTree<uchar, cv::Hamming>(data);

	tree->build();

	tree->save("test_tree.yaml.gz");
	tree->saveInvertedIndex("test_idf.yaml.gz");

	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > treeLoad =
			new cvflann::VocabTree<uchar, cv::Hamming>();

	treeLoad->load("test_tree.yaml.gz");
	treeLoad->loadInvertedIndex("test_idf.yaml.gz");

	ASSERT_TRUE(tree->size() == treeLoad->size());

	ASSERT_TRUE(*tree.obj == *treeLoad.obj);

}

TEST(VocabTree, TestDatabase) {

	/////////////////////////////////////////////////////////////////////
	cv::Mat imgDescriptors;

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_0.yaml.gz");
	keysFilenames.push_back("sift_1.yaml.gz");

	DynamicMat data(keysFilenames);
	/////////////////////////////////////////////////////////////////////

	cv::Ptr<cvflann::VocabTreeBase> tree = new cvflann::VocabTree<float,
			cv::L2<float> >(data);

	tree->build();

	tree->clearDatabase();

	bool gotException = false;
	uint i = 0;
	for (std::string keyFileName : keysFilenames) {
		try {
			imgDescriptors = cv::Mat();
			FileUtils::loadDescriptors(keyFileName, imgDescriptors);
			tree->addImageToDatabase(i, imgDescriptors);
		} catch (const std::runtime_error& error) {
			fprintf(stdout, "%s\n", error.what());
			gotException = true;
		}
		i++;
	}

	ASSERT_FALSE(gotException);

	tree->computeWordsWeights(cvflann::TF_IDF);

	tree->createDatabase();
	// TODO assert inverted files are not empty anymore

	tree->normalizeDatabase(cv::NORM_L1);
	// TODO assert DB BoW vector's values are in the [0,1] range

	// Querying the tree using the same documents used for building it
	// The top result must be the document itself and hence the score must be 1
	i = 0;
	for (std::string keyFileName : keysFilenames) {

		cv::Mat scores;

		imgDescriptors = cv::Mat();
		FileUtils::loadDescriptors(keyFileName, imgDescriptors);
		tree->scoreQuery(imgDescriptors, scores, cv::NORM_L1);
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
