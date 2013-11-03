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

#include <FileUtils.hpp>

TEST(VocabTree, Instantiation) {
	cv::Ptr<cvflann::VocabTreeBase> tree;

	EXPECT_TRUE(tree == NULL);

	tree = new cvflann::VocabTree<float, cv::L2<float> >();

	EXPECT_TRUE(tree != NULL);
}

TEST(VocabTree, LoadSaveReal) {

	// Prepare keypoints and descriptors. Recall: 128 dimensions in SIFT
	cv::Mat imgDescriptors(10, 128, cv::DataType<float>::type);
	cv::randn(imgDescriptors, cv::Scalar(128), cv::Scalar(128));
	std::vector<cv::KeyPoint> imgKeypoints;
	imgKeypoints.reserve(imgDescriptors.rows);
	imgKeypoints.insert(imgKeypoints.begin(), imgDescriptors.rows,
			cv::KeyPoint());
	FileUtils::saveFeatures("sift_descriptors.yaml.gz", imgKeypoints,
			imgDescriptors);

	std::vector<std::string> keysFilenames;
	keysFilenames.push_back("sift_descriptors.yaml.gz");
	keysFilenames.push_back("sift_descriptors.yaml.gz");

	std::vector<image> descriptorsIndices;

	int descCount = 0, imgIdx = 0;
	for (std::string keyFileName : keysFilenames) {
		// Initialize keypoints and descriptors
		// Load keypoints and descriptors
		// Check that keypoints and descriptors have same length
		for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
			image img;
			img.imgIdx = imgIdx;
			img.startIdx = descCount;
			descriptorsIndices.push_back(img);
		}
		// Increase descriptors counter
		descCount += imgDescriptors.rows;
		// Increase images counter
		imgIdx++;
	}

	DynamicMat data(descriptorsIndices, keysFilenames, descCount,
			imgDescriptors.cols, imgDescriptors.type());

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

//TEST(VocabTree, LoadSaveBinary) {
//
//	std::vector<cv::KeyPoint> imgKeypoints;
//	imgKeypoints.push_back(cv::KeyPoint());
//
//	// Number of columns is the number of bytes
//	cv::Mat imgDescriptors(3000, 32 / 8, cv::DataType<uchar>::type);
//	cv::randn(imgDescriptors, cv::Scalar(floor(UCHAR_MAX / 2)),
//			cv::Scalar(floor(UCHAR_MAX / 2)));
//
//	FileUtils::saveFeatures("binary_descriptors.yaml.gz", imgKeypoints,
//			imgDescriptors);
//
//	std::vector<std::string> keysFilenames;
//	keysFilenames.push_back("binary_descriptors.yaml.gz");
//
//	DynamicMat::image* descriptorsIndices =
//			new DynamicMat::image[keysFilenames.size()];
//
//	int descCount = 0, imgIdx = 0;
//
//	for (size_t i = 0; (int) i < imgDescriptors.rows; i++) {
//		int descriptorIndex = descCount + i;
//		descriptorsIndices[descriptorIndex].imgIdx = imgIdx;
//		descriptorsIndices[descriptorIndex].startIdx = descCount;
//		imgIdx++;
//	}
//
//	DynamicMat data(descriptorsIndices, keysFilenames, 32 / 8, 3000,
//			cv::DataType<float>::type);
//
//	imgDescriptors.release();
//
//	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > tree;
//	tree = new cvflann::VocabTree<uchar, cv::Hamming>(data);
//
//	tree->build();
//
//	tree->save("test_tree.yaml.gz");
//
//	cv::Ptr<cvflann::VocabTree<uchar, cv::Hamming> > treeLoad =
//			new cvflann::VocabTree<uchar, cv::Hamming>();
//
//	treeLoad->load("test_tree.yaml.gz");
//
//	ASSERT_TRUE(tree->size() == treeLoad->size());
//
//	ASSERT_TRUE(*tree.obj == *treeLoad.obj);
//
//}

//TEST(VocabTree, TestDatabase) {
//
//	int numDbImages = 10;
//
//	cv::Mat data(3000 * numDbImages, 128, cv::DataType<float>::type);
//	cv::RNG rng;
//	cv::randn(data, cv::Scalar(64), cv::Scalar(64));
//	rng.fill(data, cv::RNG::UNIFORM, 0, 128);
//
//	cv::Ptr<cvflann::VocabTreeBase> tree = new cvflann::VocabTree<float,
//			cv::L2<float> >(data);
//
//	tree->build();
//
//	tree->clearDatabase();
//
//	bool gotException = false;
//	for (int i = 0; i < numDbImages; i++) {
//		try {
//			CV_Assert(3000 * (i + 1) - 3000 * i);
//			tree->addImageToDatabase(i,
//					data.rowRange(cv::Range(3000 * i, 3000 * (i + 1))));
//		} catch (const std::runtime_error& error) {
//			gotException = true;
//		}
//	}
//
//	ASSERT_FALSE(gotException);
//
//	tree->computeWordsWeights(cvflann::TF_IDF);
//
//	tree->createDatabase();
//	// TODO assert inverted files are not empty anymore
//
//	tree->normalizeDatabase(cv::NORM_L1);
//	// TODO assert DB BoW vector's values are in the [0,1] range
//
//	// Querying the tree using the same documents used for building it
//	// The top result must be the document itself and hence the score must be 1
//	for (int i = 0; (int) i < numDbImages; i++) {
//		cv::Mat scores;
//
//		tree->scoreQuery(data.rowRange(cv::Range(3000 * i, 3000 * (i + 1))),
//				scores, cv::NORM_L1);
//		EXPECT_TRUE(cv::DataType<float>::type == scores.type());
//		EXPECT_TRUE(1 == scores.rows);
//		EXPECT_TRUE(data.rows == scores.cols);
//
//		cv::Mat perm;
//		cv::sortIdx(scores, perm, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);
//		EXPECT_TRUE(scores.rows == perm.rows);
//		EXPECT_TRUE(scores.cols == perm.cols);
//
//		EXPECT_TRUE(i == perm.at<int>(0, 0));
//		EXPECT_TRUE(round(scores.at<float>(0, perm.at<int>(0, 0))) == 1.0);
//	}
//
//}
