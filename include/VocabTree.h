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
#include <ScoringObject.h>
#include <CentersChooser.h>

namespace cvflann {

struct VocabTreeParams: public IndexParams {
	VocabTreeParams(int branching = 6, int depth = 10, int iterations = 11,
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
		// Cluster size
//		int size;
		// Children nodes (only for non-terminal nodes)
		VocabTreeNode** children;
		// Word id (only for terminal nodes)
		int word_id;
		// Weight (only for terminal nodes)
		double weight;
		// Assigned points (only for terminal nodes)
//		int* indices;
		VocabTreeNode() :
				center(NULL), children(NULL), word_id(-1), weight(0.0) {
		}
		VocabTreeNode& operator=(const VocabTreeNode& node) {
			center = node.center;
//			size = node.size;
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

	// The branching factor used in the hierarchical k-means clustering
	int m_branching;
	// Depth levels
	int m_depth;
	// Length of each feature.
	size_t m_veclen;

	// The root node in the tree.
	VocabTreeNodePtr m_root;
	// Pooled memory allocator
	PooledAllocator m_pool;

	// The distance
	Distance m_distance;
	// Memory occupied by the index
	int m_memoryCounter;

	// Words of the vocabulary
	std::vector<VocabTreeNodePtr> m_words;

public:

	/**
	 * Index constructor
	 *
	 * @param params - Parameters passed to the binary hierarchical k-means algorithm
	 */
	VocabTree(const cv::Mat& inputData, const IndexParams& params =
			VocabTreeParams());

	/**
	 * Index destructor, releases the memory used by the index.
	 */
	virtual ~VocabTree();

	/**
	 * Builds the index
	 *
	 * @param inputData - Matrix with the data to be clustered
	 */
	void build();

	/**
	 * Saves the tree to a stream.
	 *
	 * @param stream - The stream to save the tree to
	 */
	void save(FILE* stream) const;

	/**
	 * Loads the tree from a stream.
	 *
	 * @param stream - The stream from which the tree is loaded
	 */
	void load(FILE* stream);

	/**
	 * Returns the amount of memory (in bytes) used by the index.
	 *
	 * @return the memory used by the index
	 */
	int usedMemory() const;

	/**
	 * Quantizes a set of data into a BoW vector
	 *
	 * @param features - Matrix of data to quantize
	 * @param v - BoW vector of weighted words
	 * @param weighting - Weighting method
	 * @param scoring - Scoring method
	 */
	void quantize(const cv::Mat& features, DBoW2::BowVector &v,
			DBoW2::WeightingType weighting = DBoW2::TF_IDF,
			DBoW2::ScoringType scoring = DBoW2::L1_NORM) const;

	/**
	 * Returns the score of two vectors.
	 *
	 * @param v1 - First BoW vector
	 * @param v2 - Second BoW vector
	 * @param scoring - Scoring method
	 *
	 * @return the score between the two vectors
	 * @note v1 and v2 must be already sorted and normalized if necessary
	 */
	double score(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2,
			DBoW2::ScoringType scoring = DBoW2::L1_NORM) const;

	/**
	 * Sets the weight of the nodes of the tree according to the training data set.
	 * Before calling this function, the nodes and the words must have been already
	 * created (by calling computeClustering)
	 *
	 * @param training_data - Vector of matrices with training data
	 * @param weighting - Weighting method
	 */
	void setNodeWeights(const std::vector<cv::Mat>& training_data,
			DBoW2::WeightingType weighting = DBoW2::TF_IDF);

private:

	/**
	 * Saves the vocabulary tree starting at a given node to a stream.
	 *
	 * @param stream - The stream to save the tree to
	 * @param node - The node indicating the root of the tree to save
	 */
	void save_tree(FILE* stream, VocabTreeNodePtr node) const;

	/**
	 * Loads the vocabulary tree from a stream and stores into into a given node pointer.
	 *
	 * @param stream - The stream from which the vocabulary tree is loaded
	 * @param node - The node where to store the loaded tree
	 */
	void load_tree(FILE* stream, VocabTreeNodePtr& node);

	/**
	 * Helper function
	 */
	void free_centers(VocabTreeNodePtr node);

	/**
	 * Computes the statistics of a node (mean, radius, variance).
	 *
	 * @param node - The node to use
	 * @param indices - The array of indices of the points belonging to the node
	 * @param indices_length - The number of indices in the array of indices
	 */
	void computeNodeStatistics(VocabTreeNodePtr node, int* indices,
			int indices_length);

	/**
	 * The method responsible with actually doing the recursive hierarchical
	 * clustering.
	 *
	 * @param node - The node to cluster
	 * @param indices - Indices of the points belonging to the current node
	 * @param indices_length
	 */
	void computeClustering(VocabTreeNodePtr node, int* indices,
			int indices_length, int level);

	/**
	 * Quantizes a single data point into a word by traversing the whole tree
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
	 * @return true only if the vocabulary is empty
	 */
	bool empty() const;

	/**
	 * Creates an instance of the scoring object according to m_scoring
	 *
	 * @param scoring - Scoring method
	 * @return object for computing scores
	 */
	cv::Ptr<DBoW2::GeneralScoring> createScoringObject(
			DBoW2::ScoringType scoring = DBoW2::L1_NORM) const;

};

} /* namespace cvflann */

#endif /* BIN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
