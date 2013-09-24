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

namespace cvflann {

struct BHCIndexParams: public IndexParams {
	BHCIndexParams(int branching = 6, int depth = 10, int iterations = 11,
			flann_centers_init_t centers_init = FLANN_CENTERS_RANDOM,
			DBoW2::WeightingType weighting = DBoW2::TF_IDF,
			DBoW2::ScoringType scoring = DBoW2::L1_NORM, float cb_index = 0.2) {
		(*this)["algorithm"] = FLANN_INDEX_KMEANS;
		// branching factor
		(*this)["branching"] = branching;
		// max iterations to perform in one kmeans clustering (kmeans tree)
		(*this)["iterations"] = iterations;
		// algorithm used for picking the initial cluster centers for kmeans tree
		(*this)["centers_init"] = centers_init;
		// cluster boundary index. Used when searching the kmeans tree
		(*this)["cb_index"] = cb_index;
		// tree depth
		(*this)["depth"] = depth;
		// leafs weighting scheme
		(*this)["weighting"] = weighting;
		// BoW vectors scoring scheme
		(*this)["scoring"] = scoring;
	}
};

template<typename Distance>
class BHCIndex: public cvflann::NNIndex<Distance> {

private:

	typedef typename Distance::ElementType ElementType;

	typedef typename Distance::ResultType DistanceType;

	typedef void (BHCIndex::*centersAlgFunction)(int, int*, int, int*, int&);

	/**
	 * Structure representing a node in the hierarchical k-means tree.
	 */
	struct KMeansNode {
		// The cluster center
		uchar* center;
		// Cluster size
		int size;
		// Level
		int level;
		// Children nodes (only for non-terminal nodes)
		KMeansNode** children;
		// Word id (only for terminal nodes)
		int word_id;
		// Weight (only for terminal nodes)
		double weight;
		// Assigned points (only for terminal nodes)
		int* indices;
		KMeansNode() :
				center(NULL), size(0), level(-1), children(NULL), word_id(-1), weight(
						0.0), indices(NULL) {
		}
		KMeansNode& operator=(const KMeansNode& node) {
			center = node.center;
			size = node.size;
			level = node.level;
			children = node.children;
			word_id = node.word_id;
			weight = node.weight;
			indices = node.indices;
			return *this;
		}
	};

	typedef KMeansNode* KMeansNodePtr;

private:

	// The function used for choosing the cluster centers
	centersAlgFunction chooseCenters;
	// The branching factor used in the hierarchical k-means clustering
	int branching_;
	// Maximum number of iterations to use when performing k-means clustering
	int iterations_;
	// Algorithm for choosing the cluster centers
	flann_centers_init_t centers_init_;
	// The dataset used by this index
	const cv::Mat dataset_;
	// Index parameters
	IndexParams index_params_;
	// Number of features in the dataset
	size_t size_;
	// Length of each feature.
	size_t veclen_;
	// The root node in the tree.
	KMeansNodePtr root_;
	//  Array of indices to vectors in the dataset
	int* indices_;
	// The distance
	Distance distance_;
	// Pooled memory allocator
	PooledAllocator pool_;
	// Memory occupied by the index
	int memoryCounter_;
	// Depth levels
	int depth_;
	// Weighting method
	DBoW2::WeightingType m_weighting;
	// Scoring method
	DBoW2::ScoringType m_scoring;
	// Object for computing scores
	DBoW2::GeneralScoring* m_scoring_object;
	// Words of the vocabulary, i.e. tree leaves (m_words[wid]->word_id == wid)
	std::vector<KMeansNodePtr> m_words;

public:

	/**
	 * Index constructor
	 *
	 * @param inputData - Matrix with the data to be clustered
	 * @param params - Parameters passed to the binary hierarchical k-means algorithm
	 * @param d - The distance measure to be used
	 */
	BHCIndex(const cv::Mat& inputData, const IndexParams& params =
			BHCIndexParams(), Distance d = Distance());

	/**
	 * Index destructor, releases the memory used by the index.
	 */
	virtual ~BHCIndex();

	/**
	 * Builds the index
	 */
	void buildIndex();

	/**
	 * Saves the index to a stream.
	 *
	 * @param stream - The stream to save the index to
	 */
	void saveIndex(FILE* stream);

	/**
	 * Loads the index from a stream.
	 *
	 * @param stream - The stream from which the index is loaded
	 */
	void loadIndex(FILE* stream);

	/**
	 * Returns the number of features in this index.
	 *
	 * @return the index size
	 */
	size_t size() const;

	/**
	 * Returns the dimensionality of the features in this index.
	 *
	 * @return the index features length
	 */
	size_t veclen() const;

	/**
	 * Returns the amount of memory (in bytes) used by the index.
	 *
	 * @return the memory used by the index
	 */
	int usedMemory() const;

	/**
	 * Returns the index type (kdtree, kmeans,...)
	 *
	 * @return kmeans index type
	 */
	flann_algorithm_t getType() const;

	/**
	 * The index parameters.
	 *
	 * @return the index parameters
	 */
	IndexParams getParameters() const;

	/**
	 * Finds the nearest-neighbors to a given features vector and stores the result
	 * inside a result object.
	 *
	 * @param result - The result object in which the indices of the nearest neighbors are stored
	 * @param vec - The vector for which to search the nearest neighbors
	 * @param searchParams - Parameters than influence the search algorithm
	 */
	void findNeighbors(ResultSet<DistanceType>& result, const ElementType* vec,
			const SearchParams& searchParams);

	/**
	 * Quantizes a set of data into a BoW vector
	 *
	 * @param features - Matrix of data to quantize
	 * @param v - BoW vector of weighted words
	 */
	void quantize(const cv::Mat& features, DBoW2::BowVector &v) const;

	/**
	 * Returns the score of two vectors.
	 *
	 * @param v1 - First BoW vector
	 * @param v2 - Second BoW vector
	 * @return the score between the two vectors
	 * @note a and b must be already sorted and normalized if necessary
	 */
	inline double score(const DBoW2::BowVector &v1,
			const DBoW2::BowVector &v2) const;

private:

	/**
	 * Chooses the initial centers in the k-means clustering in a random manner.
	 *
	 * @param k - Number of centers
	 * @param indices - Vector of indices in the dataset
	 * @param indices_length - Length of indices vector
	 * @param centers - Vector of cluster centers
	 * @param centers_length - Length of centers vectors
	 */
	void chooseCentersRandom(int k, int* indices, int indices_length,
			int* centers, int& centers_length);

	/**
	 * Chooses the initial centers in the k-means using Gonzalez algorithm
	 * so that the centers are spaced apart from each other.
	 *
	 * @param k - Number of centers
	 * @param indices - Vector of indices in the dataset
	 * @param indices_length - Length of indices vector
	 * @param centers - Vector of cluster centers
	 * @param centers_length - Length of centers vectors
	 */
	void chooseCentersGonzales(int k, int* indices, int indices_length,
			int* centers, int& centers_length);

	/**
	 * Chooses the initial centers in the k-means using the k-means++ seeding
	 * algorithm proposed by Arthur and Vassilvitskii.
	 *
	 * @param k - Number of centers
	 * @param indices - Vector of indices in the dataset
	 * @param indices_length - Length of indices vector
	 * @param centers - Vector of cluster centers
	 * @param centers_length - Length of centers vectors
	 */
	void chooseCentersKMeanspp(int k, int* indices, int indices_length,
			int* centers, int& centers_length);

	/**
	 * Saves the vocabulary tree starting at a given node to a stream.
	 *
	 * @param stream - The stream to save the tree to
	 * @param node - The node indicating the root of the tree to save
	 */
	void save_tree(FILE* stream, KMeansNodePtr node);

	/**
	 * Loads the vocabulary tree from a stream and stores into into a given node pointer.
	 *
	 * @param stream - The stream from which the vocabulary tree is loaded
	 * @param node - The node where to store the loaded tree
	 */
	void load_tree(FILE* stream, KMeansNodePtr& node);

	/**
	 * Helper function
	 */
	void free_centers(KMeansNodePtr node);

	/**
	 * Computes the statistics of a node (mean, radius, variance).
	 *
	 * @param node - The node to use
	 * @param indices - The array of indices of the points belonging to the node
	 * @param indices_length - The number of indices in the array of indices
	 */
	void computeNodeStatistics(KMeansNodePtr node, int* indices,
			int indices_length);

	/**
	 * The method responsible with actually doing the recursive hierarchical
	 * clustering.
	 *
	 * @param node - The node to cluster
	 * @param indices - Indices of the points belonging to the current node
	 * @param indices_length
	 * @param branching - The branching factor to use in the clustering
	 * @param level
	 */
	//TODO for 1-sized clusters don't store a cluster center (it's the same as the single cluster point)
	void computeClustering(KMeansNodePtr node, int* indices, int indices_length,
			int branching, int level);

	/**
	 * Sets the weight of the nodes of the tree according to the training data set.
	 * Before calling this function, the nodes and the words must have been already
	 * created (by calling computeClustering)
	 *
	 * @param training_data - Vector of matrices with training data
	 */
	void setNodeWeights(const std::vector<cv::Mat>& training_data);

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
	 * @return true iff the vocabulary is empty
	 */
	inline bool empty() const;

	/**
	 * Creates an instance of the scoring object accoring to m_scoring
	 */
	void createScoringObject();
};

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::chooseCentersRandom(int k, int* indices,
		int indices_length, int* centers, int& centers_length) {
	UniqueRandom r(indices_length);

	int index;
	for (index = 0; index < k; ++index) {
		bool duplicate = true;
		int rnd;
		while (duplicate) {
			duplicate = false;
			rnd = r.next();
			if (rnd < 0) {
				centers_length = index;
				return;
			}

			centers[index] = indices[rnd];

			for (int j = 0; j < index; ++j) {
				DistanceType sq = distance_(dataset_.row(centers[index]).data,
						dataset_.row(centers[j]).data, dataset_.cols);
				if (sq < 1e-16) {
					duplicate = true;
				}
			}
		}
	}
	centers_length = index;
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::chooseCentersGonzales(int k, int* indices,
		int indices_length, int* centers, int& centers_length) {
	int n = indices_length;

	int rnd = rand_int(n);
	assert(rnd >= 0 && rnd < n);

	centers[0] = indices[rnd];

	int index;
	for (index = 1; index < k; ++index) {
		int best_index = -1;
		DistanceType best_val = 0;
		for (int j = 0; j < n; ++j) {
			DistanceType dist = distance_(dataset_.row(centers[0]).data,
					dataset_.row(indices[j]).data, dataset_.cols);
			for (int i = 1; i < index; ++i) {
				DistanceType tmp_dist = distance_(dataset_.row(centers[i]).data,
						dataset_.row(indices[j]).data, dataset_.cols);
				if (tmp_dist < dist) {
					dist = tmp_dist;
				}
			}
			if (dist > best_val) {
				best_val = dist;
				best_index = j;
			}
		}
		if (best_index != -1) {
			centers[index] = indices[best_index];
		} else {
			break;
		}
	}
	centers_length = index;
}
// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::chooseCentersKMeanspp(int k, int* indices,
		int indices_length, int* centers, int& centers_length) {
	int n = indices_length;

	double currentPot = 0;
	DistanceType* closestDistSq = new DistanceType[n];

	// Choose one random center and set the closestDistSq values
	int index = rand_int(n);
	assert(index >= 0 && index < n);
	centers[0] = indices[index];

	for (int i = 0; i < n; i++) {
		closestDistSq[i] = distance_(dataset_.row(indices[i]).data,
				dataset_.row(indices[index]).data, dataset_.cols);
		currentPot += closestDistSq[i];
	}

	const int numLocalTries = 1;

	// Choose each center
	int centerCount;
	for (centerCount = 1; centerCount < k; centerCount++) {

		// Repeat several trials
		double bestNewPot = -1;
		int bestNewIndex = -1;
		for (int localTrial = 0; localTrial < numLocalTries; localTrial++) {

			// Choose our center - have to be slightly careful to return a valid answer even accounting
			// for possible rounding errors
			double randVal = rand_double(currentPot);
			for (index = 0; index < n - 1; index++) {
				if (randVal <= closestDistSq[index])
					break;
				else
					randVal -= closestDistSq[index];
			}

			// Compute the new potential
			double newPot = 0;
			for (int i = 0; i < n; i++)
				newPot += std::min(
						distance_(dataset_.row(indices[i]).data,
								dataset_.row(indices[index]).data,
								dataset_.cols), closestDistSq[i]);

			// Store the best result
			if ((bestNewPot < 0) || (newPot < bestNewPot)) {
				bestNewPot = newPot;
				bestNewIndex = index;
			}
		}

		// Add the appropriate center
		centers[centerCount] = indices[bestNewIndex];
		currentPot = bestNewPot;
		for (int i = 0; i < n; i++)
			closestDistSq[i] = std::min(
					distance_(dataset_.row(indices[i]).data,
							dataset_.row(indices[bestNewIndex]).data,
							dataset_.cols), closestDistSq[i]);
	}

	centers_length = centerCount;

	delete[] closestDistSq;
}

// --------------------------------------------------------------------------

template<typename Distance>
BHCIndex<Distance>::BHCIndex(const cv::Mat& inputData,
		const IndexParams& params, Distance d) :
		dataset_(inputData), index_params_(params), root_(NULL), indices_(NULL), distance_(
				d), m_scoring_object(NULL) {

	// Attributes initialization
	memoryCounter_ = 0;
	size_ = dataset_.rows;
	veclen_ = dataset_.cols;

	branching_ = get_param(params, "branching", 6);
	iterations_ = get_param(params, "iterations", 11);
	depth_ = get_param(params, "depth", 10);
	m_weighting = get_param(params, "weighting", DBoW2::TF_IDF);
	m_scoring = get_param(params, "scoring", DBoW2::L1_NORM);
	createScoringObject();
	m_words.clear();

	if (iterations_ < 0) {
		iterations_ = (std::numeric_limits<int>::max)();
	}
	centers_init_ = get_param(params, "centers_init", FLANN_CENTERS_RANDOM);

	if (centers_init_ == FLANN_CENTERS_RANDOM) {
		chooseCenters = &BHCIndex::chooseCentersRandom;
	} else if (centers_init_ == FLANN_CENTERS_GONZALES) {
		chooseCenters = &BHCIndex::chooseCentersGonzales;
	} else if (centers_init_ == FLANN_CENTERS_KMEANSPP) {
		chooseCenters = &BHCIndex::chooseCentersKMeanspp;
	} else {
		throw std::runtime_error(
				"Unknown algorithm for choosing initial centers.");
	}

}

// --------------------------------------------------------------------------

template<typename Distance>
BHCIndex<Distance>::~BHCIndex() {
	if (root_ != NULL) {
		free_centers(root_);
	}
	if (indices_ != NULL) {
		delete[] indices_;
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::buildIndex() {
	if (branching_ < 2) {
		throw std::runtime_error("Branching factor must be at least 2");
	}

	indices_ = new int[size_];
	for (size_t i = 0; i < size_; ++i) {
		indices_[i] = int(i);
	}

	printf("[BHCIndex::buildIndex] Building tree from %d features\n",
			(int) size_);
	printf("[BHCIndex::buildIndex]   with depth %d, branching factor %d\n",
			depth_, branching_);
	printf("[BHCIndex::buildIndex]   and restarts %d\n", iterations_);

	root_ = pool_.allocate<KMeansNode>();
	computeNodeStatistics(root_, indices_, (int) size_);
	computeClustering(root_, indices_, (int) size_, branching_, 0);
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::saveIndex(FILE* stream) {
	// TODO Check other values to save
	save_value(stream, branching_);
	save_value(stream, iterations_);
	save_value(stream, memoryCounter_);
	save_value(stream, *indices_, (int) size_);

	save_tree(stream, root_);
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::loadIndex(FILE* stream) {
	// TODO Check other values to load
	load_value(stream, branching_);
	load_value(stream, iterations_);
	load_value(stream, memoryCounter_);
	if (indices_ != NULL) {
		delete[] indices_;
	}
	indices_ = new int[size_];
	load_value(stream, *indices_, size_);

	if (root_ != NULL) {
		free_centers(root_);
	}
	load_tree(stream, root_);

	index_params_["algorithm"] = getType();
	index_params_["branching"] = branching_;
	index_params_["iterations"] = iterations_;
	index_params_["centers_init"] = centers_init_;
}

// --------------------------------------------------------------------------

template<typename Distance>
size_t BHCIndex<Distance>::size() const {
	return size_;
}

// --------------------------------------------------------------------------

template<typename Distance>
size_t BHCIndex<Distance>::veclen() const {
	return veclen_;
}

// --------------------------------------------------------------------------

template<typename Distance>
int BHCIndex<Distance>::usedMemory() const {
	return pool_.usedMemory + pool_.wastedMemory + memoryCounter_;
}

// --------------------------------------------------------------------------

template<typename Distance>
flann_algorithm_t BHCIndex<Distance>::getType() const {
	// TODO Return the right type, clue: I stored one in the params struct
	return FLANN_INDEX_KMEANS;
}

// --------------------------------------------------------------------------

template<typename Distance>
IndexParams BHCIndex<Distance>::getParameters() const {
	return index_params_;
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::findNeighbors(ResultSet<DistanceType>& result,
		const ElementType* vec, const SearchParams& searchParams) {

	throw std::runtime_error(
			"BHCIndex::findNeighbors: error, not yet implemented method\n");

}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::save_tree(FILE* stream, KMeansNodePtr node) {
	save_value(stream, *node);
	save_value(stream, *(node->center), (int) veclen_);
	if (node->children == NULL) {
		int indices_offset = (int) (node->indices - indices_);
		save_value(stream, indices_offset);
	} else {
		for (int i = 0; i < branching_; ++i) {
			save_tree(stream, node->children[i]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::load_tree(FILE* stream, KMeansNodePtr& node) {
	node = pool_.allocate<KMeansNode>();
	load_value(stream, *node);
	node->center = new uchar[veclen_];
	load_value(stream, *(node->center), (int) veclen_);
	if (node->children == NULL) {
		int indices_offset;
		load_value(stream, indices_offset);
		node->indices = indices_ + indices_offset;
	} else {
		node->children = pool_.allocate<KMeansNodePtr>(branching_);
		for (int i = 0; i < branching_; ++i) {
			load_tree(stream, node->children[i]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::free_centers(KMeansNodePtr node) {
	delete[] node->center;
	if (node->children != NULL) {
		for (int k = 0; k < branching_; ++k) {
			free_centers(node->children[k]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::computeNodeStatistics(KMeansNodePtr node, int* indices,
		int indices_length) {

	uchar* center = new uchar[veclen_];

	memoryCounter_ += int(veclen_ * sizeof(uchar));

	// Compute center using majority voting over all data
	cv::Mat accVector(1, veclen_ * 8, cv::DataType<int>::type);
	accVector = cv::Scalar::all(0);
	for (size_t i = 0; (int) i < indices_length; i++) {
		KMajorityIndex::cumBitSum(dataset_.row(indices[i]), accVector);
	}
	cv::Mat centroid(1, veclen_, dataset_.type());
	KMajorityIndex::majorityVoting(accVector, centroid, indices_length);

	for (size_t k = 0; k < veclen_; ++k) {
		center[k] = centroid.at<uchar>(0, k);
	}

	node->center = center;
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::computeClustering(KMeansNodePtr node, int* indices,
		int indices_length, int branching, int level) {
	node->size = indices_length;
	node->level = level;

	// Recursion base case: done when the last level is reached
	// or when there are less data than clusters
	if (level == depth_ - 1 || indices_length < branching) {
		node->indices = indices;
		std::sort(node->indices, node->indices + indices_length);
		node->children = NULL;
		node->word_id = m_words.size();
		this->m_words.push_back(node);
		return;
	}

	printf("[BuildRecurse] (level %d): Running k-means (%d features)\n", level,
			indices_length);

	int* centers_idx = new int[branching];
	int centers_length;
	(this->*chooseCenters)(branching, indices, indices_length, centers_idx,
			centers_length);

	// Recursion base case: done as well if by case got
	// less cluster indices than clusters
	if (centers_length < branching) {
		node->indices = indices;
		std::sort(node->indices, node->indices + indices_length);
		node->children = NULL;
		node->word_id = m_words.size();
		this->m_words.push_back(node);
		delete[] centers_idx;
		return;
	}

	// TODO initCentroids: assign centers based on the chosen indexes

	printf(
			"[BuildRecurse] (level %d): initCentroids - assign centers based on the chosen indexes\n",
			level);

	cv::Mat dcenters(branching, veclen_, dataset_.type());
	for (int i = 0; i < centers_length; i++) {
		dataset_.row(centers_idx[i]).copyTo(
				dcenters(cv::Range(i, i + 1), cv::Range(0, veclen_)));
	}
	delete[] centers_idx;

	std::vector<DistanceType> radiuses(branching);
	int* count = new int[branching];
	for (int i = 0; i < branching; ++i) {
		radiuses[i] = 0;
		count[i] = 0;
	}

	//TODO quantize: assign points to clusters
	int* belongs_to = new int[indices_length];
	for (int i = 0; i < indices_length; ++i) {

		DistanceType sq_dist = distance_(dataset_.row(indices[i]).data,
				dcenters.row(0).data, veclen_);
		belongs_to[i] = 0;
		for (int j = 1; j < branching; ++j) {
			DistanceType new_sq_dist = distance_(dataset_.row(indices[i]).data,
					dcenters.row(j).data, veclen_);
			if (sq_dist > new_sq_dist) {
				belongs_to[i] = j;
				sq_dist = new_sq_dist;
			}
		}
		count[belongs_to[i]]++;
	}

	bool converged = false;
	int iteration = 0;
	while (!converged && iteration < iterations_) {
		converged = true;
		iteration++;

		//TODO: computeCentroids compute the new cluster centers
		// Warning: using matrix of integers, there might be an overflow when summing too much descriptors
		cv::Mat bitwiseCount(branching, veclen_ * 8, cv::DataType<int>::type);
		// Zeroing matrix of cumulative bits
		bitwiseCount = cv::Scalar::all(0);
		// Zeroing all the centroids dimensions
		dcenters = cv::Scalar::all(0);

		// Bitwise summing the data into each centroid
		for (size_t i = 0; (int) i < indices_length; i++) {
			uint j = belongs_to[i];
			cv::Mat b = bitwiseCount.row(j);
			KMajorityIndex::cumBitSum(dataset_.row(indices[i]), b);
		}
		// Bitwise majority voting
		for (size_t j = 0; (int) j < branching; j++) {
			cv::Mat centroid = dcenters.row(j);
			KMajorityIndex::majorityVoting(bitwiseCount.row(j), centroid,
					count[j]);
		}

		// TODO quantize: reassign points to clusters
		for (int i = 0; i < indices_length; ++i) {
			DistanceType sq_dist = distance_(dataset_.row(indices[i]).data,
					dcenters.row(0).data, veclen_);
			int new_centroid = 0;
			for (int j = 1; j < branching; ++j) {
				DistanceType new_sq_dist = distance_(
						dataset_.row(indices[i]).data, dcenters.row(j).data,
						veclen_);
				if (sq_dist > new_sq_dist) {
					new_centroid = j;
					sq_dist = new_sq_dist;
				}
			}
			if (new_centroid != belongs_to[i]) {
				count[belongs_to[i]]--;
				count[new_centroid]++;
				belongs_to[i] = new_centroid;

				converged = false;
			}
		}

		// TODO handle empty clusters
		for (int i = 0; i < branching; ++i) {
			// if one cluster converges to an empty cluster,
			// move an element into that cluster
			if (count[i] == 0) {
				int j = (i + 1) % branching;
				while (count[j] <= 1) {
					j = (j + 1) % branching;
				}

				for (int k = 0; k < indices_length; ++k) {
					if (belongs_to[k] == j) {
						belongs_to[k] = i;
						count[j]--;
						count[i]++;
						break;
					}
				}
				converged = false;
			}
		}

	}

	printf("[BuildRecurse] (level %d): Finished clustering\n", level);

	uchar** centers = new uchar*[branching];

	for (int i = 0; i < branching; ++i) {
		centers[i] = new uchar[veclen_];
		memoryCounter_ += (int) (veclen_ * sizeof(uchar));
		for (size_t k = 0; k < veclen_; ++k) {
			centers[i][k] = dcenters.at<uchar>(i, k);
		}
	}

	// compute kmeans clustering for each of the resulting clusters
	node->children = pool_.allocate<KMeansNodePtr>(branching);
	int start = 0;
	int end = start;
	for (int c = 0; c < branching; ++c) {
		// Re-order indices by chunks in clustering order
		for (int i = 0; i < indices_length; ++i) {
			if (belongs_to[i] == c) {
				std::swap(indices[i], indices[end]);
				std::swap(belongs_to[i], belongs_to[end]);
				end++;
			}
		}

		node->children[c] = pool_.allocate<KMeansNode>();
		node->children[c]->center = centers[c];
		node->children[c]->indices = NULL;
		computeClustering(node->children[c], indices + start, end - start,
				branching, level + 1);
		start = end;
	}

	dcenters.release();
	delete[] centers;
	delete[] count;
	delete[] belongs_to;
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::setNodeWeights(
		const std::vector<cv::Mat>& training_matrices) {
	const uint NWords = m_words.size();
	const uint NDocs = training_matrices.size();

	if (m_weighting == DBoW2::TF || m_weighting == DBoW2::BINARY) {
		// IDF part must be 1 always
		for (size_t i = 0; i < NWords; i++) {
			m_words[i]->weight = 1;
		}
	} else if (m_weighting == DBoW2::IDF || m_weighting == DBoW2::TF_IDF) {
		// IDF and TF-IDF: we calculate the IDF path now

		// Note: this actually calculates the IDF part of the TF-IDF score.
		// The complete TF-IDF score is calculated in ::transform

		// Ni: number of documents/images in which the ith words appears
		// TODO When test see if this is the same as the inverted file length,
		// for me it looks like it does
		std::vector<uint> Ni(NWords, 0);
		std::vector<bool> counted(NWords, false);

		for (cv::Mat training_data : training_matrices) {
			// Restart word count 'cause new image features matrix
			std::fill(counted.begin(), counted.end(), false);
			for (size_t i = 0; i < training_data.rows; i++) {
				uint word_id;
//				transform(*fit, word_id);
				// Count only once the appearance of the word in the image (training matrix)
				if (!counted[word_id]) {
					Ni[word_id]++;
					counted[word_id] = true;
				}
			}
		}

		// Set ln(N/Ni)
		for (size_t i = 0; i < NWords; i++) {
			if (Ni[i] > 0) {
				m_words[i]->weight = log((double) NDocs / (double) Ni[i]);
			}
			// TODO else: this cannot occur if using kmeans++
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::quantize(const cv::Mat& features,
		DBoW2::BowVector &v) const {

	if (features.type() != CV_8U) {
		throw std::runtime_error(
				"BHCIndex::quantize: error, features matrix is not binary\n");
	}

	if (features.cols != (int) veclen_) {
		std::stringstream msg;
		msg << "BHCIndex::quantize: error, features vectors must be " << veclen_
				<< " bytes long, that is " << veclen_ * 8 << "-dimensional\n";
		throw std::runtime_error(msg.str());
	}

	if (features.rows < 1) {
		throw std::runtime_error(
				"BHCIndex::quantize: error, need at least one feature vector to quantize\n");
	}

	v.clear();

	if (empty()) {
		return;
	}

	// normalize
	DBoW2::LNorm norm;
	bool must = m_scoring_object->mustNormalize(norm);

	if (m_weighting == DBoW2::TF || m_weighting == DBoW2::TF_IDF) {
		for (size_t i = 0; (int) i < features.rows; i++) {
			uint id;
			double w;
			// w is the IDF value if TF_IDF, 1 if TF

			quantize(features.row(i), id, w);

			// not stopped
			if (w > 0) {
				v.addWeight(id, w);
			}
		}

		if (!v.empty() && !must) {
			// unnecessary when normalizing
			const double nd = v.size();

			for (DBoW2::BowVector::iterator vit = v.begin(); vit != v.end();
					vit++) {
				vit->second /= nd;
			}
		}
	} else // IDF or BINARY
	{
		for (size_t i = 0; (int) i < features.rows; i++) {
			uint id;
			double w;
			// w is the inverse document frequency if IDF, or 1 if BINARY
			quantize(features.row(i), id, w);
			// not stopped
			if (w > 0) {
				v.addIfNotExist(id, w);
			}
		} // if add_features
	} // if m_weighting == ...

	if (must) {
		v.normalize(norm);
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::quantize(const cv::Mat &features, uint &word_id,
		double &weight) const {

	KMeansNodePtr best_node = root_;

	while (best_node->children != NULL) {

		KMeansNodePtr node = best_node;

		// Arbitrarily assign to first child
		best_node = node->children[0];
		DistanceType best_distance = distance_(features.data, best_node->center,
				veclen_);

		// Looking for a better child
		for (size_t j = 1; (int) j < this->branching_; j++) {
			DistanceType d = distance_(features.data, node->children[j]->center,
					veclen_);
			if (d < best_distance) {
				best_distance = d;
				best_node = node->children[j];
			}
		}
	}

	// Turn node id into word id
	word_id = best_node->word_id;
	weight = best_node->weight;
}

// --------------------------------------------------------------------------

template<typename Distance>
inline bool BHCIndex<Distance>::empty() const {
	return m_words.empty();
}

// --------------------------------------------------------------------------

template<typename Distance>
inline double BHCIndex<Distance>::score(const DBoW2::BowVector &v1,
		const DBoW2::BowVector &v2) const {
	return m_scoring_object->score(v1, v2);
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::createScoringObject() {
	delete m_scoring_object;
	m_scoring_object = NULL;

	switch (m_scoring) {
	case DBoW2::L1_NORM:
		m_scoring_object = new DBoW2::L1Scoring;
		break;

	case DBoW2::L2_NORM:
		m_scoring_object = new DBoW2::L2Scoring;
		break;

	case DBoW2::CHI_SQUARE:
		m_scoring_object = new DBoW2::ChiSquareScoring;
		break;

	case DBoW2::KL:
		m_scoring_object = new DBoW2::KLScoring;
		break;

	case DBoW2::BHATTACHARYYA:
		m_scoring_object = new DBoW2::BhattacharyyaScoring;
		break;

	case DBoW2::DOT_PRODUCT:
		m_scoring_object = new DBoW2::DotProductScoring;
		break;

	}
}

} /* namespace cvflann */
#endif /* BIN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
