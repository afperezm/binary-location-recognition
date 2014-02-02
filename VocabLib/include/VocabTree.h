/***********************************************************************
 *
 * VocabTree.h
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

#ifndef VOCAB_TREE_H_
#define VOCAB_TREE_H_

#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/flann/flann.hpp>

#include <CentersChooser.h>
#include <DirectIndex.hpp>
#include <DynamicMat.hpp>
#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <KMajority.h>
#include <InvertedIndex.hpp>

#include <fstream>

namespace vlr {

enum WeightingType {
	TF_IDF, TF, BINARY
};

// --------------------------------------------------------------------------

struct VocabTreeParams: public cvflann::IndexParams {
	VocabTreeParams(int branching = 10, int depth = 7, int iterations = 10,
			cvflann::flann_centers_init_t centers_init =
					cvflann::FLANN_CENTERS_RANDOM, int levels_up = 2) {
		// branching factor
		(*this)["branching"] = branching;
		// max iterations to perform in one k-means clustering
		(*this)["iterations"] = iterations;
		// tree depth
		(*this)["depth"] = depth;
		// algorithm used for picking the initial cluster centers for k-means tree
		(*this)["centers_init"] = centers_init;
		// Levels to go up the tree to select nodes to store in the direct index
		(*this)["levels_up"] = levels_up;
	}
};

// --------------------------------------------------------------------------

class VocabTreeBase {
public:

	/**
	 * Virtual destroyer to enable destruction from a subclass.
	 */
	virtual ~VocabTreeBase() {
	}

	virtual void build() = 0;

	virtual void quantize(const cv::Mat& feature, int diLevel, int& wordId,
			int& nodeAtL) const = 0;

	virtual void save(const std::string& filename) const = 0;

	virtual void load(const std::string& filename) = 0;

	virtual size_t size() = 0;

	virtual size_t getWordsCount() const = 0;

	virtual int getDepth() const = 0;

	virtual size_t getVeclen() const = 0;

};

// --------------------------------------------------------------------------

/**
 * Structure representing a node in the hierarchical k-means tree.
 */
template<typename TDescriptor>
struct VocabTreeNode {
	// The node id
	int node_id;
	// The cluster center
	TDescriptor* center;
	// Children nodes (only for non-terminal nodes)
	// Note: no need to store how many children does it has because
	// this is a k-ary tree, where 'k' is the branch factor, i.e. it has k children or none
	VocabTreeNode** children;
	// Word id (only for terminal nodes)
	int word_id;
	VocabTreeNode() :
			node_id(-1), center(NULL), children(NULL), word_id(-1) {
	}
//		VocabTreeNode& operator=(const VocabTreeNode& node) {
//			node_id = node.node_id;
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
//			word_id = node.word_id;
//			return *this;
//		}
};

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
class VocabTree: public VocabTreeBase {

private:

	typedef typename Distance::ResultType DistanceType;
	typedef VocabTreeNode<TDescriptor>* VocabTreeNodePtr;

protected:

	/** Attributes useful for building the tree **/
	// The function used for choosing the cluster centers
	cvflann::flann_centers_init_t m_centers_init;
	// Maximum number of iterations to use when performing k-means clustering
	int m_iterations;
	// The data set used by this index
	vlr::Mat& m_dataset;

	/** Attributes of the tree **/
	// Branching factor (number of partitions in which
	// data is divided at each level of the tree)
	int m_branching;
	// Number of levels of the tree
	int m_depth;
	// Length of each feature vector
	size_t m_veclen;
	// Number of nodes in the tree
	size_t m_size;
	// The root node of the tree
	VocabTreeNodePtr m_root;
	// Words of the vocabulary
	std::vector<VocabTreeNodePtr> m_words;

	/** Other attributes **/
	// The distance measure used to evaluate similarity between features
	Distance m_distance;

public:

	/**
	 * Class constructor.
	 *
	 * @inputData - Reference to the matrix with the data to be clustered
	 * @param params - Parameters to the hierarchical k-means algorithm
	 */
	VocabTree(vlr::Mat& inputData = vlr::DEFAULT_INPUTDATA,
			const VocabTreeParams& params = VocabTreeParams());

	/**
	 * Class destroyer, releases the memory used by the tree.
	 */
	virtual ~VocabTree();

	/**
	 * Builds the tree.
	 *
	 * @note After this method is executed m_root holds a pointer to the tree,
	 *		 while m_words holds pointers to the leaf nodes.
	 * @note Interior nodes have only 'center' and 'children' information,
	 * 		 while leaf nodes have only 'center' and 'word_id', all weights for
	 * 		 interior nodes are 0 while weights for leaf nodes are 1.
	 */
	void build();

	void quantize(const cv::Mat& feature, int diLevel, int& wordId,
			int& nodeAtL) const;

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

	/**
	 * Returns the tree size.
	 *
	 * @return number of nodes in the tree
	 */
	size_t size();

	size_t getWordsCount() const {
		return m_words.size();
	}

	bool operator==(const VocabTree<TDescriptor, Distance> &other) const;

	bool operator!=(const VocabTree<TDescriptor, Distance> &other) const;

	/**** Getters ****/
	VocabTreeNodePtr getRoot() const {
		return m_root;
	}

	int getBranching() const {
		return m_branching;
	}

	int getDepth() const {
		return m_depth;
	}

	size_t getVeclen() const {
		return m_veclen;
	}

private:

	/**
	 * Recursively releases the memory allocated to store the tree node centers.
	 *
	 * @param node - A pointer to a node in the tree where to start the releasing
	 */
	void free_centers(VocabTreeNodePtr node);

	/**
	 * The method responsible with actually doing the recursive hierarchical clustering.
	 *
	 * @param node - The node to cluster
	 * @param indices - Indices of the points belonging to the current node
	 * @param indices_length
	 */
	void computeClustering(VocabTreeNodePtr node, int* indices,
			int indices_length, int level, bool fitted);

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
	void load_tree(boost::iostreams::filtering_istream& is,
			VocabTreeNodePtr& node);

	/**
	 * Returns whether the tree is empty.
	 *
	 * @return true if and only if the tree is empty
	 */
	bool empty() const;

	/**
	 *
	 * @param a
	 * @param b
	 * @return
	 */
	bool compareEqual(const VocabTreeNodePtr a, const VocabTreeNodePtr b) const;

	/**
	 * Copy constructor and the assignment operator are private
	 * to prevent obtaining copies of the instance.
	 */
	VocabTree(VocabTree const&); // Don't Implement
	void operator=(VocabTree const&); // Don't implement

};

// --------------------------------------------------------------------------

typedef VocabTree<float, cv::L2<float> > VocabTreeReal;
typedef VocabTree<uchar, cv::Hamming> VocabTreeBin;

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
VocabTree<TDescriptor, Distance>::VocabTree(vlr::Mat& inputData,
		const VocabTreeParams& params) :
		m_dataset(inputData), m_veclen(0), m_size(0), m_root(NULL), m_distance(
				Distance()) {

	// Attributes initialization
	m_veclen = m_dataset.cols;
	m_branching = cvflann::get_param(params, "branching", 10);
	m_iterations = cvflann::get_param(params, "iterations", 10);
	m_depth = cvflann::get_param(params, "depth", 7);
	m_centers_init = cvflann::get_param(params, "centers_init",
			cvflann::FLANN_CENTERS_RANDOM);

	if (m_iterations < 0) {
		m_iterations = std::numeric_limits<int>::max();
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
VocabTree<TDescriptor, Distance>::~VocabTree() {
	if (m_root != NULL) {
		free_centers(m_root);
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::free_centers(VocabTreeNodePtr node) {
	delete[] node->center;
	if (node->children != NULL) {
		for (int k = 0; k < m_branching; ++k) {
			free_centers(node->children[k]);
		}
		delete[] node->children;
	}
	delete node;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
size_t VocabTree<TDescriptor, Distance>::size() {
	return m_size;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::build() {

	if (m_branching < 2) {
		throw std::runtime_error("[VocabTree::build] Error, branching factor"
				" must be at least 2");
	}

	if (m_depth < 2) {
		throw std::runtime_error("[VocabTree::build] Error, depth"
				" must be at least 2");
	}

	if (m_dataset.empty() == true) {
		throw std::runtime_error("[VocabTree::build] Error, data set is empty"
				" cannot proceed with clustering");
	}

	// Number of features in the data set
	int size = m_dataset.rows;

	//  Array of descriptors indices
	int* indices = new int[size];
	for (int i = 0; i < size; ++i) {
		indices[i] = i;
	}

	m_root = new VocabTreeNode<TDescriptor>();
	m_root->center = new TDescriptor[m_veclen];
	std::fill(m_root->center, m_root->center + m_veclen, 0);

#if VTREEVERBOSE
	printf("[VocabTree::build] Started clustering\n");
#endif

	computeClustering(m_root, indices, size, 0, false);

#if VTREEVERBOSE
	printf("[VocabTree::build] Finished clustering\n");
#endif

	delete[] indices;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::quantize(const cv::Mat& feature,
		int diLevel, int& wordId, int& nodeAtL) const {

	CV_Assert(0 <= diLevel && diLevel < m_depth);

	VocabTreeNodePtr best_node = m_root;

	int level = 0;

	while (best_node->children != NULL) {

		VocabTreeNodePtr node = best_node;

		// Arbitrarily assign to first child
		best_node = node->children[0];
		DistanceType best_distance = m_distance((TDescriptor*) feature.data,
				best_node->center, m_veclen);

		// Looking for a better child
		for (int j = 1; j < m_branching; ++j) {
			DistanceType d = m_distance((TDescriptor*) feature.data,
					node->children[j]->center, m_veclen);
			if (d < best_distance) {
				best_distance = d;
				best_node = node->children[j];
				if (level == diLevel) {
					nodeAtL = j;
				}
			}
		}

		++level;
	}

	wordId = best_node->word_id;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::save(const std::string& filename) const {

	if (empty()) {
		throw std::runtime_error("[VocabTree::save] Tree is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error(
				"[VocabTree::save] Error opening file " + filename
						+ " for writing");
	}

	fs << "iterations" << m_iterations;
	fs << "branching" << m_branching;
	fs << "depth" << m_depth;
	fs << "vectorLength" << (int) m_veclen;
	fs << "size" << (int) m_size;

	fs << "nodes" << "[";

	save_tree(fs, m_root);

	fs << "]";

	fs.release();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::save_tree(cv::FileStorage& fs,
		VocabTreeNodePtr node) const {

	// Save node
	fs << "{";
	fs << "center"
			<< cv::Mat(1, m_veclen, cv::DataType<TDescriptor>::type,
					(uchar*) node->center);
	fs << "nodeId" << node->node_id;
	fs << "wordId" << node->word_id;
	fs << "}";

	// Save children, if any
	if (node->children != NULL) {
		for (int i = 0; i < m_branching; ++i) {
			save_tree(fs, node->children[i]);
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::load(const std::string& filename) {

	std::ifstream inputZippedFileStream;
	boost::iostreams::filtering_istream inputFileStream;

	std::string line, field;
	std::stringstream ss;

	enum treeFields {
		iterations, branching, depth, vectorLength, size, nodes
	};
	std::string treeFieldsNames[] = { "iterations:", "branching:", "depth:",
			"vectorLength:", "size:", "nodes:" };

	// Open file
	inputZippedFileStream.open(filename.c_str(),
			std::fstream::in | std::fstream::binary);

	// Check file
	if (inputZippedFileStream.good() == false) {
		throw std::runtime_error("[VocabTree::load] "
				"Unable to open file [" + filename + "] for reading");
	}

	try {
		inputFileStream.push(boost::iostreams::gzip_decompressor());
		inputFileStream.push(inputZippedFileStream);

		while (getline(inputFileStream, line)) {
			ss.clear();
			ss.str(line);
			ss >> field;
			if (field.compare(treeFieldsNames[iterations]) == 0) {
				ss >> m_iterations;
			} else if (field.compare(treeFieldsNames[branching]) == 0) {
				ss >> m_branching;
			} else if (field.compare(treeFieldsNames[depth]) == 0) {
				ss >> m_depth;
			} else if (field.compare(treeFieldsNames[vectorLength]) == 0) {
				ss >> m_veclen;
			} else if (field.compare(treeFieldsNames[size]) == 0) {
				ss >> m_size;
			} else if (field.compare(treeFieldsNames[nodes]) == 0) {
				break;
			}
		}

		m_root = new VocabTreeNode<TDescriptor>();
		load_tree(inputFileStream, m_root);

	} catch (const boost::iostreams::gzip_error& e) {
		throw std::runtime_error("[VocabTree::load] "
				"Got error while parsing file [" + std::string(e.what()) + "]");
	}

	// Close file
	inputZippedFileStream.close();

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::load_tree(
		boost::iostreams::filtering_istream& inputFileStream,
		VocabTreeNodePtr& node) {

	enum nodeFields {
		start, center, rows, cols, dt, data, nodeId, wordId
	};
	std::string nodeFieldsNames[] = { "-", "center:", "rows:", "cols:", "dt:",
			"data:", "nodeId:", "wordId:" };

	std::string line, field;
	std::stringstream ss;

	cv::Mat _center;

	int _rows = -1;
	int _cols = -1;
	std::string _type;
	int colIdx = -1;
	float elem;

	while (getline(inputFileStream, line)) {
		ss.clear();
		ss.str(line);
		ss >> field;
		if (field.compare(nodeFieldsNames[start]) == 0) {
			continue;
		} else if (field.compare(nodeFieldsNames[center]) == 0) {
			continue;
		} else if (field.compare(nodeFieldsNames[rows]) == 0) {
			ss >> _rows;
		} else if (field.compare(nodeFieldsNames[cols]) == 0) {
			ss >> _cols;
		} else if (field.compare(nodeFieldsNames[dt]) == 0) {
			ss >> _type;
		} else if (field.compare(nodeFieldsNames[nodeId]) == 0) {
			ss >> node->node_id;
		} else if (field.compare(nodeFieldsNames[wordId]) == 0) {
			ss >> node->word_id;
			break;
		} else {
			if (field.compare(nodeFieldsNames[data]) == 0) {
				_center = cv::Mat::zeros(_rows, _cols,
						_type.compare("f") == 0 ? CV_32F : CV_8U);
				line.replace(line.find(nodeFieldsNames[data]), 5, " ");
			}

			bool isLastLine = line.find("]") != std::string::npos;

			std::replace(line.begin(), line.end(), '[', ' ');
			std::replace(line.begin(), line.end(), ',', ' ');
			std::replace(line.begin(), line.end(), ']', ' ');

			ss.clear();
			ss.str(line);

			while ((ss >> elem).fail() == false) {
				_center.at<TDescriptor>(0, ++colIdx) = elem;
			}

			if (isLastLine) {
				// Check dimensions correctness
				CV_Assert(_center.rows == 1);
				CV_Assert(_center.cols == int(m_veclen));

				// Deep copy
				node->center = new TDescriptor[m_veclen];
				for (size_t k = 0; k < m_veclen; ++k) {
					node->center[k] = _center.at<TDescriptor>(0, k);
				}

				// Release memory and dereference header
				_center.release();
				_center = cv::Mat();
			}
		}
	}

	bool hasChildren = node->word_id == -1;

	if (hasChildren == false) {
		// Node has no children then it's a leaf node
		node->children = NULL;
		m_words.push_back(node);
	} else {
		// Node has children then it's an interior node
		node->children = new VocabTreeNodePtr[m_branching];
		for (int c = 0; c < m_branching; ++c) {
			node->children[c] = new VocabTreeNode<TDescriptor>();
			load_tree(inputFileStream, node->children[c]);
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::computeClustering(VocabTreeNodePtr node,
		int* indices, int indices_length, int level, bool fitted) {

	node->node_id = m_size;
	++m_size;

	// Sort descriptors, caching leverages this fact
	// Note: it doesn't affect the clustering process since all descriptors referenced by indices belong to the same cluster
	if (level > 0) {
		std::sort(indices, indices + indices_length);
	}

	// Recursion base case: done when the last level is reached
	// or when there is less data than clusters
	if (level == m_depth - 1 || indices_length < m_branching) {
		node->children = NULL;
		node->word_id = m_words.size();
		m_words.push_back(node);
#if VTREEVERBOSE
		printf(
				"[VocabTree::computeClustering] (level %d): last level was reached or there was less data than clusters (%d features)\n",
				level, indices_length);
#endif
		return;
	}

#if VTREEVERBOSE
	printf(
			"[VocabTree::computeClustering] (level %d): Running k-means (%d features)\n",
			level, indices_length);
#endif

	std::vector<int> centers_idx(m_branching);
	int centers_length;

#if DEBUG
#if VTREEVERBOSE
	printf("randomCenters - Start\n");
#endif
#endif

	CentersChooser<TDescriptor, Distance>::create(m_centers_init)->chooseCenters(
			m_branching, indices, indices_length, centers_idx, centers_length,
			m_dataset);

#if DEBUG
#if VTREEVERBOSE
	printf("randomCenters - End\n");
#endif
#endif

	// Recursion base case: done as well if by case got
	// less cluster indices than clusters
#ifdef SUPPDUPLICATES
	if (centers_length < m_branching) {
		node->children = NULL;
		node->word_id = m_words.size();
		m_words.push_back(node);
#if VTREEVERBOSE
		printf(
				"[VocabTree::computeClustering] (level %d): got less cluster indices than clusters (%d features)\n",
				level, indices_length);
#endif
		return;
	}
#else
	CV_Assert(centers_length == m_branching);
#endif

#if DEBUG
#if VTREEVERBOSE
	printf("initCentroids - Start\n");
#endif
#endif

	cv::Mat dcenters(m_branching, m_veclen, m_dataset.type());
	for (int i = 0; i < centers_length; ++i) {
		m_dataset.row(centers_idx[i]).copyTo(
				dcenters(cv::Range(i, i + 1), cv::Range(0, m_veclen)));
	}

#if DEBUG
#if VTREEVERBOSE
	printf("initCentroids - End\n");
#endif
#endif

	std::vector<int> count(m_branching);
	for (int i = 0; i < m_branching; ++i) {
		count[i] = 0;
	}

	// Prepare cache for clustering, clear it if descriptors
	// didn't fit in memory at previous level but they do at this one
	if (fitted == false && indices_length <= m_dataset.getCapacity()) {
		m_dataset.clearCache();
		fitted = true;
#if VTREEVERBOSE
		printf("Clearing cache at level=[%d]\n", level);
#endif
	}

#if DEBUG
#if VTREEVERBOSE
	printf("quantize - Start\n");
#endif
#endif

	std::vector<int> belongs_to(indices_length);
	for (int i = 0; i < indices_length; ++i) {
		DistanceType sq_dist = m_distance(
				(TDescriptor*) m_dataset.row(indices[i]).data,
				(TDescriptor*) dcenters.row(0).data, m_veclen);
		belongs_to[i] = 0;
		for (int j = 1; j < m_branching; ++j) {
			DistanceType new_sq_dist = m_distance(
					(TDescriptor*) m_dataset.row(indices[i]).data,
					(TDescriptor*) dcenters.row(j).data, m_veclen);
			if (sq_dist > new_sq_dist) {
				belongs_to[i] = j;
				sq_dist = new_sq_dist;
			}
		}
		++count[belongs_to[i]];
	}

#if DEBUG
#if VTREEVERBOSE
	printf("quantize - End\n");
#endif
#endif

	bool converged = false;
	int iteration = 0;
	while (converged == false && iteration < m_iterations) {
#if DEBUG
#if VTREEVERBOSE
		printf("iteration=[%d]\n", iteration);
#endif
#endif

		converged = true;
		++iteration;

#if DEBUG
#if VTREEVERBOSE
		printf("computeCentroids - Start\n");
#endif
#endif

		// Zeroing all the centroids dimensions
		dcenters = cv::Scalar::all(0);

		if (m_dataset.type() == CV_8U) {
			// Warning: using matrix of integers, there might be
			// an overflow when summing too much descriptors
			cv::Mat bitwiseCount(m_branching, m_veclen * 8,
					cv::DataType<int>::type);
			// Zeroing matrix of cumulative bits
			bitwiseCount = cv::Scalar::all(0);
			// Bitwise summing the data into each centroid
			for (size_t i = 0; (int) i < indices_length; ++i) {
				uint j = belongs_to[i];
				cv::Mat b = bitwiseCount.row(j);
				KMajority::cumBitSum(m_dataset.row(indices[i]), b);
			}
			// Bitwise majority voting
			for (size_t j = 0; (int) j < m_branching; ++j) {
				cv::Mat centroid = dcenters.row(j);
				KMajority::majorityVoting(bitwiseCount.row(j), centroid,
						count[j]);
			}
		} else {
			// Accumulate data into its corresponding cluster accumulator
			for (size_t i = 0; (int) i < indices_length; ++i) {
				for (size_t k = 0; k < m_veclen; ++k) {
					dcenters.at<TDescriptor>(belongs_to[i], k) += m_dataset.row(
							indices[i]).at<TDescriptor>(0, k);
				}
			}
			// Divide accumulated data by the number transaction assigned to the cluster
			for (size_t i = 0; (int) i < m_branching; ++i) {
				for (size_t k = 0; k < m_veclen; ++k) {
					dcenters.at<TDescriptor>(i, k) /= count[i];
				}
			}
		}
#if DEBUG
#if VTREEVERBOSE
		printf("computeCentroids - End\n");
#endif
#endif

#if DEBUG
#if VTREEVERBOSE
		printf("quantize - Start\n");
#endif
#endif

		for (int i = 0; i < indices_length; ++i) {
			DistanceType sq_dist = m_distance(
					(TDescriptor*) m_dataset.row(indices[i]).data,
					(TDescriptor*) dcenters.row(0).data, m_veclen);
			int new_centroid = 0;
			for (int j = 1; j < m_branching; ++j) {
				DistanceType new_sq_dist = m_distance(
						(TDescriptor*) m_dataset.row(indices[i]).data,
						(TDescriptor*) dcenters.row(j).data, m_veclen);
				if (sq_dist > new_sq_dist) {
					new_centroid = j;
					sq_dist = new_sq_dist;
				}
			}
			if (new_centroid != belongs_to[i]) {
				--count[belongs_to[i]];
				++count[new_centroid];
				belongs_to[i] = new_centroid;

				converged = false;
			}
		}

#if DEBUG
#if VTREEVERBOSE
		printf("quantize - End\n");
#endif
#endif

#if DEBUG
#if VTREEVERBOSE
		printf("handleEmptyClusters - Start\n");
#endif
#endif

		// TODO Handle empty clusters

#if DEBUG
#if VTREEVERBOSE
		printf("handleEmptyClusters - End\n");
#endif
#endif

	}

	TDescriptor** centers = new TDescriptor*[m_branching];

	for (int i = 0; i < m_branching; ++i) {
		centers[i] = new TDescriptor[m_veclen];
		for (size_t k = 0; k < m_veclen; ++k) {
			centers[i][k] = dcenters.at<TDescriptor>(i, k);
		}
	}

	// Compute k-means clustering for each of the resulting clusters
	node->children = new VocabTreeNodePtr[m_branching];
	int start = 0;
	int end = start;
	for (int c = 0; c < m_branching; ++c) {

#if VTREEVERBOSE
		printf(
				"[VocabTree::computeClustering] Clustering over resulting clusters, level=[%d] branch=[%d]\n",
				level, c);
#endif

		// Re-order indices by chunks in clustering order
		for (int i = 0; i < indices_length; ++i) {
			if (belongs_to[i] == c) {
				std::swap(indices[i], indices[end]);
				std::swap(belongs_to[i], belongs_to[end]);
				++end;
			}
		}

		node->children[c] = new VocabTreeNode<TDescriptor>();
		node->children[c]->center = centers[c];
		computeClustering(node->children[c], indices + start, end - start,
				level + 1, fitted);
		start = end;
	}

	dcenters.release();
	delete[] centers;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
bool VocabTree<TDescriptor, Distance>::empty() const {
	return m_size == 0;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
bool VocabTree<TDescriptor, Distance>::compareEqual(const VocabTreeNodePtr a,
		const VocabTreeNodePtr b) const {

#if DEBUG
#if VTREEVERBOSE
	printf("[VocabTree::compareEqual] Comparing tree roots\n");
#endif
#endif

	// Assert both nodes are interior or leaf nodes
	if ((a->children != NULL && b->children == NULL)
			|| (a->children == NULL && b->children != NULL)) {
		return false;
	}

	// At this point both nodes have none or some children,
	// hence valid nodes so we proceed to check the centers
	for (size_t k = 0; k < m_veclen; ++k) {
		// Might be necessary to check as well the type of pointer
		// i.e. both should be pointers to the same type
		if (a->center[k] != b->center[k]) {
			return false;
		}
	}

	if (a->children == NULL) {
		// Base case: both are leaf nodes since have no children
		return true;
	} else {
		// Recursion case: both are interior nodes
		for (size_t i = 0; (int) i < m_branching; ++i) {
			if (compareEqual(a->children[i], b->children[i]) == false) {
				return false;
			}
		}
	}

	return true;
}

template<class TDescriptor, class Distance>
bool VocabTree<TDescriptor, Distance>::operator==(
		const VocabTree<TDescriptor, Distance> &other) const {

	if (getVeclen() != other.getVeclen()
			|| getBranching() != other.getBranching()
			|| getDepth() != other.getDepth()) {
#if DEBUG
#if VTREEVERBOSE
		printf("[VocabTree::operator==] Vector length, branch factor or depth are not equal\n");
#endif
#endif
		return false;
	}

	if (compareEqual(getRoot(), other.getRoot()) == false) {
#if DEBUG
#if VTREEVERBOSE
		printf("[VocabTree::operator==] Tree is not equal\n");
#endif
#endif
		return false;
	}

	return true;
}

template<class TDescriptor, class Distance>
bool VocabTree<TDescriptor, Distance>::operator!=(
		const VocabTree<TDescriptor, Distance> &other) const {
	return !(*this == other);
}

} /* namespace vlr */

#endif /* VOCAB_TREE_H_ */
