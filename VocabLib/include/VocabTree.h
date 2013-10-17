/***********************************************************************
 *
 * bin_hierarchical_clustering_index.h
 *
 *  Created on: Sep 18, 2013
 *      Author: andresf
 *
 ***********************************************************************
 *
 * Software License Agreement (BSD License)
 *
 * Copyright 2008-2009  Marius Muja (mariusm@cs.ubc.ca). All rights reserved.
 * Copyright 2008-2009  David G. Lowe (lowe@cs.ubc.ca). All rights reserved.
 *
 * THE BSD LICENSE
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *************************************************************************/

#ifndef BIN_HIERARCHICAL_CLUSTERING_INDEX_H_
#define BIN_HIERARCHICAL_CLUSTERING_INDEX_H_

#include <opencv2/flann/flann.hpp>
#include <CentersChooser.h>

namespace cvflann {

enum WeightingType
{
  TF_IDF,
  BINARY
};

struct VocabTreeParams: public IndexParams {
	VocabTreeParams(int branching = 10, int depth = 6, int iterations = 11,
			flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM) {
		// branching factor
		(*this)["branching"] = branching;
		// max iterations to perform in one kmeans clustering (kmeans tree)
		(*this)["iterations"] = iterations;
		// tree depth
		(*this)["depth"] = depth;
		// algorithm used for picking the initial cluster centers for kmeans tree
		(*this)["centers_init"] = centers_init;
	}
};

class ImageCount {
public:
	ImageCount() :
			m_index(0), m_count(0.0) {
	}
	ImageCount(unsigned int index, float count) :
			m_index(index), m_count(count) {
	}

	// Index of the database image this entry corresponds to
	unsigned int m_index;
	// (Weighted, normalized) Count of how many times this feature appears
	float m_count;
};

class VocabTree {

private:

	typedef uchar TDescriptor;
	typedef cv::flann::Hamming<TDescriptor> Distance;
	typedef typename Distance::ElementType ElementType;
	typedef typename Distance::ResultType DistanceType;

	/**
	 * Structure representing a node in the hierarchical k-means tree.
	 */
	struct VocabTreeNode {
		// The cluster center
		TDescriptor* center;
		// Children nodes (only for non-terminal nodes)
		// Note: no need to store how many children does it has because
		// this is a k-ary tree, where 'k' is the branch factor, that is has k children or none
		VocabTreeNode** children;
		// Word id (only for terminal nodes)
		// TODO Check if this attribute is truly necessary
		int word_id;
		// Weight (only for terminal nodes)
		double weight;
		// Inverse document/image list (only for terminal nodes)
		std::vector<ImageCount> image_list;
		VocabTreeNode() :
				center(NULL), children(NULL), word_id(-1), weight(0.0) {
			image_list.clear();
		}
		VocabTreeNode& operator=(const VocabTreeNode& node) {
			center = node.center;
			children = node.children;
			word_id = node.word_id;
			weight = node.weight;
			return *this;
		}
	};

	typedef VocabTreeNode* VocabTreeNodePtr;

protected:

	/* Attributes useful for clustering */
	// The function used for choosing the cluster centers
	cvflann::flann_centers_init_t m_centers_init;
	// Maximum number of iterations to use when performing k-means clustering
	int m_iterations;
	// The dataset used by this index
	const cv::Mat& m_dataset;

	/* Attributes useful for describing the tree */
	// The branching factor used in the hierarchical k-means clustering
	int m_branching;
	// Depth levels
	int m_depth;
	// Length of each feature.
	size_t m_veclen;

	/* Attributes actually holding the tree */
	// The root node in the tree.
	VocabTreeNodePtr m_root;
	// Words of the vocabulary
	std::vector<VocabTreeNodePtr> m_words;

	/* Attributes used by several methods */
	// The distance
	Distance m_distance;
	// Memory occupied by the index
	int m_memoryCounter;

public:

	/**
	 * Tree constructor
	 *
	 * @param params - Parameters passed to the binary hierarchical k-means algorithm
	 */
	VocabTree(const cv::Mat& inputData = cv::Mat(), const IndexParams& params =
			VocabTreeParams());

	/**
	 * Tree destructor, releases the memory used by the tree.
	 */
	virtual ~VocabTree();

	/**
	 * Returns the size of the tree.
	 *
	 * @return number of leaf nodes in the tree
	 */
	size_t size();

	/**
	 * Builds the tree.
	 *
	 * @param inputData - Matrix with the data to be clustered
	 *
	 * @note After this method is executed m_root holds a pointer to the tree,
	 *		 while m_words holds pointers to the leaf nodes.
	 * @note Interior nodes have only 'center' and 'children' information,
	 * 		 while leaf nodes have only 'center' and 'word_id', all weights for
	 * 		 interior nodes are 0 while weights for leaf nodes are 1.
	 */
	void build();

	/**
	 * Saves the tree to a stream.
	 *
	 * @param stream - The stream to save the tree to
	 */
	void save(const std::string& filename) const;

	/**
	 * Loads the tree from a stream.
	 *
	 * @param stream - The stream from which the tree is loaded
	 */
	void load(const std::string& filename);

	/**
	 * Pushes DB image features down the tree until a leaf node,
	 * once reached updates the inverted file.
	 *
	 * @param imgIdx - The id of the image
	 * @param imgFeatures - Matrix of features representing the image
	 */
	void addImageToDatabase(uint imgIdx, cv::Mat dbImgFeatures);

	/**
	 * Using the pre-loaded inverted files assigns words weights
	 * according to the chosen weighting scheme.
	 *
	 * @param NWords
	 * @param weighting - The weighting scheme to apply
	 */
	void computeWordsWeights(WeightingType weighting,
			const uint numDbWords = 0);

	/**
	 * Computes the DB BoW vectors by applying the words weights
	 * to the image counts in the inverted files.
	 *
	 * @note Might be better not computing the DB BoW vectors in advance
	 *		 but simply holding the histogram counts and obtaining
	 *		 the score by component-wise weighting and scoring (DRY way)
	 */
	void createDatabase();

	/**
	 * Normalizes the DB BoW vectors by dividing the weighted counts
	 * stored in the inverted files.
	 *
	 * @param numDbImages
	 * @param normType
	 */
	void normalizeDatabase(const uint numDbImages, int normType = cv::NORM_L1);

	/**
	 * Clears the inverted files from the leaf nodes
	 */
	void clearDatabase();

	/**
	 * Computes the query BoW vector of an image by pushing down the tree the query image
	 * features and applying the words weights, followed by efficiently scoring it against
	 * the pre-computed DB BoW vectors.
	 *
	 * @param queryImgFeatures - Matrix containing the features of the query image
	 * @param numDbImages - Number of DB images, used for creating the scores matrix
	 * @param scores - Row matrix of size [1 x n] where n is the number DB images
	 * @param scoringMethod - normalization method used for scoring BoW vectors
	 *
	 * @note DB BoW vectors must be normalized beforehand
	 */
	void scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
			const uint numDbImages, const int normType = cv::NORM_L2) const;

private:

	/**
	 * Recursively releases the memory allocated to store the tree node centers.
	 *
	 * @param node - A pointer to a node in the tree where to start the releasing
	 */
	void free_centers(VocabTreeNodePtr node);

	/**
	 * Computes the centroid of a node. (Only for the root node)
	 *
	 * @param node - The node to use
	 * @param indices - The array of indices of the points belonging to the node
	 * @param indices_length - The number of indices in the array of indices
	 */
	void computeNodeStatistics(VocabTreeNodePtr node, int* indices,
			int indices_length);

	/**
	 * The method responsible with actually doing the recursive hierarchical clustering.
	 *
	 * @param node - The node to cluster
	 * @param indices - Indices of the points belonging to the current node
	 * @param indices_length
	 */
	void computeClustering(VocabTreeNodePtr node, int* indices,
			int indices_length, int level);

	/**
	 * Saves the vocabulary tree starting at a given node to a stream.
	 *
	 * @param fs - A reference to the file storage pointing to the file where to save the tree
	 * @param node - The node indicating the root of the tree to save
	 */
	void save_tree(cv::FileStorage& fs, VocabTreeNodePtr node) const;

	/**
	 * Loads the vocabulary tree from a stream and stores into into a given node pointer.
	 *
	 * @param filename - A reference to the file storage where to read node parameters
	 * @param node - The node where to store the loaded tree
	 */
	void load_tree(cv::FileNode& fs, VocabTreeNodePtr& node);

	/**
	 * Quantizes a single feature vector into a word. Traverses the whole tree,
	 * and stores the resulting word id and weight.
	 *
	 * @param feature - Row vector representing the feature vector to quantize
	 * @param id - The id of the found word
	 * @param weight - The weight of the found word
	 */
	void quantize(const cv::Mat& feature, uint &id, double &weight) const;

	/**
	 * Returns whether the vocabulary is empty (i.e. it has not been trained)
	 *
	 * @return true if and only if the vocabulary is empty
	 */
	bool empty() const;

	/**
	 * Updates the inverted file of the given word by adding the image indicated
	 * by the given imgIdx.
	 *
	 * @param wordIdx - The id of the word whose inverted file to update
	 * @param imgIdx - The id of the image to add to the inverted file
	 *
	 * @note Images are added in sequence
	 */
	void addFeatureToInvertedFile(uint wordIdx, uint imgIdx);

	/**
	 * Transforms a set of data (representing a single image) into a BoW vector.
	 *
	 * @param featuresVector - Matrix of data to quantize
	 * @param bowVector - BoW vector of weighted words
	 * @param normType - Norm used to normalize the output query BoW vector
	 */
	void transform(const cv::Mat& featuresVector, cv::Mat& bowVector,
			const int& normType) const;

};

} /* namespace cvflann */

#endif /* BIN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
