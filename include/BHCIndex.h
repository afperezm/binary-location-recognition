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
		// The cluster size (number of points in the cluster)
		int size;
		// Child nodes (only for non-terminal nodes)
		KMeansNode** childs;
		// Node points (only for terminal nodes)
		int* indices;
		// Level
		int level;

		// Node id
		DBoW2::NodeId id;
		// Weight if the node is a word
		DBoW2::WordValue weight;
		// Parent node id (undefined in case of root)
		DBoW2::NodeId parent;
		// The cluster center
		cv::Mat descriptor;
//		DistanceType* pivot;
		// Word id if the node is a word
		DBoW2::WordId word_id;

		KMeansNode() :
				size(0), childs(new KMeansNode[0]), indices(new int[0]), level(
						-1), id(0), weight(0.0), parent(-1), descriptor(
						cv::Mat()), word_id(-1) {
		}

	};

	typedef KMeansNode* KMeansNodePtr;

	typedef BranchStruct<KMeansNodePtr, DistanceType> BranchSt;

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
	std::vector<cv::Ptr<KMeansNode> > m_words;

public:

	/**
	 * Index constructor
	 *
	 * @param inputData - Matrix with the data to be clustered
	 * @param params - Parameters passed to the binary hierarchical k-means algorithm
	 * @param d
	 */
	BHCIndex(const cv::Mat& inputData, const IndexParams& params =
			BHCIndexParams(), Distance d = Distance()) :
			dataset_(inputData), index_params_(params), root_(NULL), indices_(
					NULL), distance_(d), m_scoring_object(NULL) {

		// Attributes initialization
		memoryCounter_ = 0;
		size_ = dataset_.rows;
		veclen_ = dataset_.cols;

		branching_ = get_param(params, "branching", 6);
		iterations_ = get_param(params, "iterations", 11);
		depth_ = get_param(params, "depth", 10);
		m_weighting = get_param(params, "weighting", DBoW2::TF_IDF);
		m_scoring = get_param(params, "scoring", DBoW2::L1_NORM);
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

	/**
	 * Index destructor.
	 *
	 * Release the memory used by the index.
	 */
	virtual ~BHCIndex();

	/**
	 * Builds the index
	 */
	void buildIndex();

	void saveIndex(FILE* stream);

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

//	/**
//	 * Clustering function that takes a cut in the hierarchical k-means
//	 * tree and return the clusters centers of that clustering.
//	 * Params:
//	 *     numClusters = number of clusters to have in the clustering computed
//	 * Returns: number of cluster centers
//	 */
//	int getClusterCenters(Matrix<DistanceType>& centers) {
//		int numClusters = centers.rows;
//		if (numClusters < 1) {
//			throw FLANNException("Number of clusters must be at least 1");
//		}
//
//		DistanceType variance;
//		KMeansNodePtr* clusters = new KMeansNodePtr[numClusters];
//
//		int clusterCount = getMinVarianceClusters(root_, clusters, numClusters,
//				variance);
//
//		Logger::info("Clusters requested: %d, returning %d\n", numClusters,
//				clusterCount);
//
//		for (int i = 0; i < clusterCount; ++i) {
//			DistanceType* center = clusters[i]->pivot;
//			for (size_t j = 0; j < veclen_; ++j) {
//				centers[i][j] = center[j];
//			}
//		}
//		delete[] clusters;
//
//		return clusterCount;
//	}

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

	void save_tree(FILE* stream, KMeansNodePtr node);

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

//	/**
//	 * Performs one descent in the hierarchical k-means tree. The branches not
//	 * visited are stored in a priority queue.
//	 *
//	 * Params:
//	 *      node = node to explore
//	 *      result = container for the k-nearest neighbors found
//	 *      vec = query points
//	 *      checks = how many points in the dataset have been checked so far
//	 *      maxChecks = maximum dataset points to checks
//	 */
//	void findNN(KMeansNodePtr node, ResultSet<DistanceType>& result,
//			const ElementType* vec, int& checks, int maxChecks,
//			Heap<BranchSt>* heap) {
//		// Ignore those clusters that are too far away
//		{
//			DistanceType bsq = distance_(vec, node->pivot, veclen_);
//			DistanceType rsq = node->radius;
//			DistanceType wsq = result.worstDist();
//
//			DistanceType val = bsq - rsq - wsq;
//			DistanceType val2 = val * val - 4 * rsq * wsq;
//
//			//if (val>0) {
//			if ((val > 0) && (val2 > 0)) {
//				return;
//			}
//		}
//
//		if (node->childs == NULL) {
//			if (checks >= maxChecks) {
//				if (result.full())
//					return;
//			}
//			checks += node->size;
//			for (int i = 0; i < node->size; ++i) {
//				int index = node->indices[i];
//				DistanceType dist = distance_(dataset_[index], vec, veclen_);
//				result.addPoint(dist, index);
//			}
//		} else {
//			DistanceType* domain_distances = new DistanceType[branching_];
//			int closest_center = exploreNodeBranches(node, vec,
//					domain_distances, heap);
//			delete[] domain_distances;
//			findNN(node->childs[closest_center], result, vec, checks, maxChecks,
//					heap);
//		}
//	}
//
//	/**
//	 * Helper function that computes the nearest childs of a node to a given query point.
//	 * Params:
//	 *     node = the node
//	 *     q = the query point
//	 *     distances = array with the distances to each child node.
//	 * Returns:
//	 */
//	int exploreNodeBranches(KMeansNodePtr node, const ElementType* q,
//			DistanceType* domain_distances, Heap<BranchSt>* heap) {
//
//		int best_index = 0;
//		domain_distances[best_index] = distance_(q,
//				node->childs[best_index]->pivot, veclen_);
//		for (int i = 1; i < branching_; ++i) {
//			domain_distances[i] = distance_(q, node->childs[i]->pivot, veclen_);
//			if (domain_distances[i] < domain_distances[best_index]) {
//				best_index = i;
//			}
//		}
//
//		//		float* best_center = node->childs[best_index]->pivot;
//		for (int i = 0; i < branching_; ++i) {
//			if (i != best_index) {
//				domain_distances[i] -= cb_index_ * node->childs[i]->variance;
//
//				//				float dist_to_border = getDistanceToBorder(node.childs[i].pivot,best_center,q);
//				//				if (domain_distances[i]<dist_to_border) {
//				//					domain_distances[i] = dist_to_border;
//				//				}
//				heap->insert(BranchSt(node->childs[i], domain_distances[i]));
//			}
//		}
//
//		return best_index;
//	}
//
//	/**
//	 * Function the performs exact nearest neighbor search by traversing the entire tree.
//	 */
//	void findExactNN(KMeansNodePtr node, ResultSet<DistanceType>& result,
//			const ElementType* vec) {
//		// Ignore those clusters that are too far away
//		{
//			DistanceType bsq = distance_(vec, node->pivot, veclen_);
//			DistanceType rsq = node->radius;
//			DistanceType wsq = result.worstDist();
//
//			DistanceType val = bsq - rsq - wsq;
//			DistanceType val2 = val * val - 4 * rsq * wsq;
//
//			//                  if (val>0) {
//			if ((val > 0) && (val2 > 0)) {
//				return;
//			}
//		}
//
//		if (node->childs == NULL) {
//			for (int i = 0; i < node->size; ++i) {
//				int index = node->indices[i];
//				DistanceType dist = distance_(dataset_[index], vec, veclen_);
//				result.addPoint(dist, index);
//			}
//		} else {
//			int* sort_indices = new int[branching_];
//
//			getCenterOrdering(node, vec, sort_indices);
//
//			for (int i = 0; i < branching_; ++i) {
//				findExactNN(node->childs[sort_indices[i]], result, vec);
//			}
//
//			delete[] sort_indices;
//		}
//	}
//
//	/**
//	 * Helper function.
//	 *
//	 * I computes the order in which to traverse the child nodes of a particular node.
//	 */
//	void getCenterOrdering(KMeansNodePtr node, const ElementType* q,
//			int* sort_indices) {
//		DistanceType* domain_distances = new DistanceType[branching_];
//		for (int i = 0; i < branching_; ++i) {
//			DistanceType dist = distance_(q, node->childs[i]->pivot, veclen_);
//
//			int j = 0;
//			while (domain_distances[j] < dist && j < i)
//				j++;
//			for (int k = i; k > j; --k) {
//				domain_distances[k] = domain_distances[k - 1];
//				sort_indices[k] = sort_indices[k - 1];
//			}
//			domain_distances[j] = dist;
//			sort_indices[j] = i;
//		}
//		delete[] domain_distances;
//	}
//
//	/**
//	 * Method that computes the squared distance from the query point q
//	 * from inside region with center c to the border between this
//	 * region and the region with center p
//	 */
//	DistanceType getDistanceToBorder(DistanceType* p, DistanceType* c,
//			DistanceType* q) {
//		DistanceType sum = 0;
//		DistanceType sum2 = 0;
//
//		for (int i = 0; i < veclen_; ++i) {
//			DistanceType t = c[i] - p[i];
//			sum += t * (q[i] - (c[i] + p[i]) / 2);
//			sum2 += t * t;
//		}
//
//		return sum * sum / sum2;
//	}
//
//	/**
//	 * Helper function the descends in the hierarchical k-means tree by spliting those clusters that minimize
//	 * the overall variance of the clustering.
//	 * Params:
//	 *     root = root node
//	 *     clusters = array with clusters centers (return value)
//	 *     varianceValue = variance of the clustering (return value)
//	 * Returns:
//	 */
//	int getMinVarianceClusters(KMeansNodePtr root, KMeansNodePtr* clusters,
//			int clusters_length, DistanceType& varianceValue) {
//		int clusterCount = 1;
//		clusters[0] = root;
//
//		DistanceType meanVariance = root->variance * root->size;
//
//		while (clusterCount < clusters_length) {
//			DistanceType minVariance =
//					(std::numeric_limits<DistanceType>::max)();
//			int splitIndex = -1;
//
//			for (int i = 0; i < clusterCount; ++i) {
//				if (clusters[i]->childs != NULL) {
//
//					DistanceType variance = meanVariance
//							- clusters[i]->variance * clusters[i]->size;
//
//					for (int j = 0; j < branching_; ++j) {
//						variance += clusters[i]->childs[j]->variance
//								* clusters[i]->childs[j]->size;
//					}
//					if (variance < minVariance) {
//						minVariance = variance;
//						splitIndex = i;
//					}
//				}
//			}
//
//			if (splitIndex == -1)
//				break;
//			if ((branching_ + clusterCount - 1) > clusters_length)
//				break;
//
//			meanVariance = minVariance;
//
//			// split node
//			KMeansNodePtr toSplit = clusters[splitIndex];
//			clusters[splitIndex] = toSplit->childs[0];
//			for (int i = 1; i < branching_; ++i) {
//				clusters[clusterCount++] = toSplit->childs[i];
//			}
//		}
//
//		varianceValue = meanVariance / root->size;
//		return clusterCount;
//	}

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

	int maxChecks = get_param(searchParams, "checks", 32);

	if (maxChecks == FLANN_CHECKS_UNLIMITED) {
//			findExactNN(root_, result, vec);
	} else {
		// Priority queue storing intermediate branches in the best-bin-first search
		Heap<BranchSt>* heap = new Heap<BranchSt>((int) size_);

		int checks = 0;
//			findNN(root_, result, vec, checks, maxChecks, heap);

		BranchSt branch;
		while (heap->popMin(branch) && (checks < maxChecks || !result.full())) {
			KMeansNodePtr node = branch.node;
//				findNN(node, result, vec, checks, maxChecks, heap);
		}
		assert(result.full());

		delete heap;
	}

}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::save_tree(FILE* stream, KMeansNodePtr node) {
	save_value(stream, *node);
//	save_value(stream, *(node->pivot), (int) veclen_);
	if (node->childs == NULL) {
		int indices_offset = (int) (node->indices - indices_);
		save_value(stream, indices_offset);
	} else {
		for (int i = 0; i < branching_; ++i) {
			save_tree(stream, node->childs[i]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::load_tree(FILE* stream, KMeansNodePtr& node) {
	node = pool_.allocate<KMeansNode>();
	load_value(stream, *node);
//	node->pivot = new DistanceType[veclen_];
//	load_value(stream, *(node->pivot), (int) veclen_);
	if (node->childs == NULL) {
		int indices_offset;
		load_value(stream, indices_offset);
		node->indices = indices_ + indices_offset;
	} else {
		node->childs = pool_.allocate<KMeansNodePtr>(branching_);
		for (int i = 0; i < branching_; ++i) {
			load_tree(stream, node->childs[i]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::free_centers(KMeansNodePtr node) {
//	delete[] node->pivot;
	if (node->childs != NULL) {
		for (int k = 0; k < branching_; ++k) {
			free_centers(node->childs[k]);
		}
	}
}

// --------------------------------------------------------------------------

template<typename Distance>
void BHCIndex<Distance>::computeNodeStatistics(KMeansNodePtr node, int* indices,
		int indices_length) {

	DistanceType radius = 0;
	DistanceType variance = 0;
	DistanceType* mean = new DistanceType[veclen_];
	memoryCounter_ += int(veclen_ * sizeof(DistanceType));

	memset(mean, 0, veclen_ * sizeof(DistanceType));

//	node->pivot = mean;
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
		node->childs = NULL;
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
		node->childs = NULL;
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
		for (unsigned int i = 0; (int) i < indices_length; i++) {
			unsigned int j = belongs_to[i];
			// Finding all data assigned to jth clusther
			uchar byte = 0;
			for (int l = 0; l < bitwiseCount.cols; l++) {
				// bit: 7-(l%8) col: (int)l/8 descriptor: i
				// Load byte every 8 bits
				if ((l % 8) == 0) {
					byte = *(dataset_.row(i).col((int) l / 8).data);
				}
				// Warning: ignore maybe-uninitialized warning because loop starts with l=0 that means byte gets a value as soon as the loop start
				// bit at lth position is mod(bitleftshift(byte,i),2) where lth position is 7-mod(l,8) i.e 7, 6, 5, 4, 3, 2, 1, 0
				bitwiseCount.at<int>(j, l) += ((int) ((byte >> (7 - (l % 8)))
						% 2));
			}
		}
		// Bitwise majority voting
		for (unsigned int j = 0; (int) j < branching; j++) {
			// In this point I already have stored in bitwiseCount the bitwise sum of all data assigned to jth cluster
			for (int l = 0; l < bitwiseCount.cols; l++) {
				// If the bitcount for jth cluster at dimension l is greater than half of the data assigned to it
				// then set lth centroid bit to 1 otherwise set it to 0 (break ties randomly)
				bool bit;
				// There is a tie if the number of data assigned to jth cluster is even
				// AND the number of bits set to 1 in lth dimension is the half of the data assigned to jth cluster
				if (count[j] % 2 == 1
						&& 2 * bitwiseCount.at<int>(j, l) == (int) count[j]) {
					bit = rand() % 2;
				} else {
					bit = 2 * bitwiseCount.at<int>(j, l) > (int) (count[j]);
				}
				dcenters.at<unsigned char>(j,
						(int) (bitwiseCount.cols - 1 - l) / 8) += (bit)
						<< ((bitwiseCount.cols - 1 - l) % 8);
			}
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

	DistanceType** centers = new DistanceType*[branching];

	for (int i = 0; i < branching; ++i) {
		centers[i] = new DistanceType[veclen_];
		memoryCounter_ += (int) (veclen_ * sizeof(DistanceType));
		for (size_t k = 0; k < veclen_; ++k) {
			centers[i][k] = (DistanceType) dcenters.at<uchar>(i, k);
		}
	}

	// compute kmeans clustering for each of the resulting clusters
	node->childs = pool_.allocate<KMeansNodePtr>(branching);
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

		node->childs[c] = pool_.allocate<KMeansNode>();
		node->childs[c]->descriptor = dcenters.row(c);
		node->childs[c]->indices = NULL;
		computeClustering(node->childs[c], indices + start, end - start,
				branching, level + 1);
		start = end;
	}

	dcenters.release();
	delete[] centers;
	delete[] count;
	delete[] belongs_to;
}

// --------------------------------------------------------------------------

} /* namespace cvflann */
#endif /* BIN_HIERARCHICAL_CLUSTERING_INDEX_H_ */
