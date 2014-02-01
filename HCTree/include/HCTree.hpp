/*
 * HCTree.hpp
 *
 *  Created on: Jan 20, 2014
 *      Author: andresf
 */

#ifndef HCMTREE_HPP_
#define HCMTREE_HPP_

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>

#include <CentersChooser.h>
#include <DynamicMat.hpp>

#include <stdlib.h>

namespace vlr {

typedef uchar TDescriptor;
typedef cv::Hamming Distance;
typedef typename Distance::ResultType DistanceType;

struct HCTreeParams: public cvflann::IndexParams {
	HCTreeParams(int branching = 10, int maxLeafSize = 7) {
		// Branching factor
		(*this)["branching"] = branching;
		// Maximum leaf size
		(*this)["maxLeafSize"] = maxLeafSize;
	}
};

class HCTree {

private:

	/**
	 * Structure representing a node in the hierarchical k-means tree.
	 */
	struct HCTreeNode {
		// The node id
		int nodeId;
		// The cluster center
		TDescriptor* center;
		// Children nodes (only for non-terminal nodes)
		HCTreeNode** children;
		// Empty constructor
		HCTreeNode() :
				nodeId(-1), center(NULL), children(NULL) {
		}
//		// Assignment operator
//		HCTreeNode& operator=(const HCTreeNode& node) {
//			nodeId = node.nodeId;
//			if (node.center != NULL) {
//				// Deep copy
//				center = new TDescriptor[m_veclen];
//				for (size_t k = 0; k < m_veclen; ++k) {
//					center[k] = node.center[k];
//				}
//			} else {
//				center = NULL;
//			}
//			children = node.children;
//			return *this;
//		}
	};

	typedef HCTreeNode* HCTreeNodePtr;

protected:

	/** Attributes useful for building the tree **/
	// The data set over which to build the tree
	vlr::Mat& m_dataset;

	/** Attributes of the tree **/
	// Branching factor (number of partitions in which
	// data is divided at each level of the tree)
	int m_branching;
	// Threshold on the number of points inside a cluster
	// to consider a node as a leaf
	int m_maxLeafSize;
	// Length of each feature
	size_t m_veclen;
	// Number of nodes in the tree
	size_t m_size;
	// The root node of the tree
	HCTreeNodePtr m_root;

	/** Other attributes **/
	Distance m_distance;

public:

	/**
	 * Class constructor.
	 *
	 * @param inputData - Reference to the matrix with the data to be clustered.
	 * @param params - Parameters to the hierarchical clustering tree
	 */
	HCTree(vlr::Mat& inputData = DEFAULT_INPUTDATA,
			const HCTreeParams& params = HCTreeParams());

	/**
	 * Class destroyer, releases the memory used by the tree.
	 */
	virtual ~HCTree();

	/**
	 * Returns the tree size.
	 *
	 * @return number of nodes in the tree
	 */
	size_t size() const;

	/**
	 * Returns the dimensionality of the data points being clustered.
	 *
	 * @return number of dimensions of clustering data
	 */
	size_t getVeclen() const {
		return m_veclen;
	}

	int getBranching() const {
		return m_branching;
	}

	int getMaxLeafSize() const {
		return m_maxLeafSize;
	}

	HCTreeNodePtr getRoot() const {
		return m_root;
	}

	/**
	 * Returns whether the tree is empty.
	 *
	 * @return true if and only if the tree is empty
	 */
	bool empty() const;

	/**
	 * Builds the tree.
	 */
	void build();

	/**
	 * Strict equality operator.
	 *
	 * @param other
	 * @return true if objects are equal, false otherwise
	 */
	bool operator==(const HCTree &other) const;

	/**
	 * Strict inequality operator.
	 *
	 * @param other
	 * @return true if objects are unequal, false otherwise
	 */
	bool operator!=(const HCTree &other) const;

	/**
	 * Saves the tree to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the tree
	 */
	void save(const std::string& filename) const;

	/**
	 * Loads the tree from a file stream.
	 *
	 * @param filename - The name of the file stream from where to load the tree
	 */
	void load(const std::string& filename);

private:

	/**
	 * Recursively releases the memory allocated to store the tree node centers.
	 *
	 * @param node - A pointer to a node in the tree where to start the releasing
	 */
	void free_centers(HCTreeNodePtr node);

	/**
	 * Responsible with actually doing the recursive hierarchical clustering.
	 *
	 * @param node - The node to cluster
	 * @param indices - Indices of the points belonging to the current node
	 * @param indices_length
	 * @param level
	 * @param fitted
	 */
	void computeClustering(HCTreeNodePtr node, int* indices, int indices_length,
			int level, bool fitted);

	/**
	 * Saves to a stream the tree starting at a given node.
	 *
	 * @param fs - A reference to the file storage pointing to the file where to save the tree
	 * @param node - The node indicating the root of the tree to save
	 */
	void save_tree(cv::FileStorage& fs, HCTreeNodePtr node) const;

	/**
	 * Loads the tree from a stream and stores it into a given node pointer.
	 *
	 * @param filename - A reference to the file storage where to read node parameters
	 * @param node - The node where to store the loaded tree
	 */
	void load_tree(boost::iostreams::filtering_istream& is,
			HCTreeNodePtr& node);

	bool compareEqual(const HCTreeNodePtr a, const HCTreeNodePtr b) const;

	// Make private the copy constructor and the assignment operator
	// to prevent obtaining copies of the instance
	HCTree(HCTree const&); // Don't Implement
	void operator=(HCTree const&); // Don't implement

};

} /* namespace vlr */

#endif /* HCMTREE_HPP_ */
