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

#include <opencv2/core/core_c.h>
#include <opencv2/flann/flann.hpp>

#include <CentersChooser.h>
#include <DirectIndex.hpp>
#include <DynamicMat.hpp>
#include <FunctionUtils.hpp>
#include <KMajorityIndex.h>

namespace bfeat {

enum WeightingType {
	TF_IDF, BINARY
};

struct VocabTreeParams: public cvflann::IndexParams {
	VocabTreeParams(int branching = 10, int depth = 6, int iterations = 1,
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

class VocabTreeBase {
public:
	// Virtual destroyer to enable destruction from a subclass
	virtual ~VocabTreeBase() {
	}

	virtual size_t size() = 0;

	virtual void build() = 0;

	virtual void save(const std::string& filename) const = 0;

	virtual void load(const std::string& filename) = 0;

	virtual void saveInvertedIndex(const std::string& filename) const = 0;

	virtual void loadInvertedIndex(const std::string& filename) = 0;

	virtual void saveDirectIndex(const std::string& filename) const = 0;

	virtual void loadDirectIndex(const std::string& filename) = 0;

	virtual void quantize(const cv::Mat& feature, uint &id, double &weight,
			int &nodeAtL) const=0;

	virtual void addImageToDatabase(uint imgIdx, cv::Mat dbImgFeatures) = 0;

	virtual void computeWordsWeights(WeightingType weighting) = 0;

	virtual void createDatabase() = 0;

	virtual void normalizeDatabase(int normType = cv::NORM_L1) = 0;

	virtual void clearDatabase() = 0;

	virtual void scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
			const int normType = cv::NORM_L2) const = 0;

	virtual void getDbBoWVector(uint idx, cv::Mat& dbBowVector) const = 0;

	virtual int getWordId(size_t i) const = 0;

	virtual double getWordWeight(size_t i) const = 0;

	virtual const std::vector<ImageCount>& getWordImageList(size_t i) const = 0;

	virtual int getDirectIndexLevel() = 0;

	virtual void setDirectIndexLevel(int levelsUp) = 0;

	virtual int getDepth() const = 0;
};

static DynamicMat DEFAULT_INPUTDATA = DynamicMat();

template<class TDescriptor, class Distance>
class VocabTree: public VocabTreeBase {

private:

	typedef typename Distance::ResultType DistanceType;

	/**
	 * Structure representing a node in the hierarchical k-means tree.
	 */
	struct VocabTreeNode {
		// The node id
		int node_id;
		// The cluster center
		TDescriptor* center;
		// Children nodes (only for non-terminal nodes)
		// Note: no need to store how many children does it has because
		// this is a k-ary tree, where 'k' is the branch factor, that is has k children or none
		VocabTreeNode** children;
		// Word id (only for terminal nodes)
		int word_id;
		// Weight (only for terminal nodes)
		double weight;
		// Inverse document/image list (only for terminal nodes)
		std::vector<ImageCount> image_list;
		VocabTreeNode() :
				node_id(-1), center(NULL), children(NULL), word_id(-1), weight(
						0.0) {
		}
		VocabTreeNode& operator=(const VocabTreeNode& node) {
			node_id = node.node_id;
			if (node.center != NULL) {
				// Deep copy
				center = new TDescriptor[m_veclen];
				for (size_t k = 0; k < m_veclen; ++k) {
					center[k] = node.center[k];
				}
			} else {
				center = NULL;
			}
			children = node.children;
			word_id = node.word_id;
			weight = node.weight;
			return *this;
		}
	};

	typedef VocabTreeNode* VocabTreeNodePtr;

	typedef std::vector<int> FeatureVector;
	typedef std::map<int, FeatureVector> TreeNode;
	typedef std::vector<TreeNode> DirectIndex;

protected:

	/** Attributes useful for clustering **/
	// The function used for choosing the cluster centers
	cvflann::flann_centers_init_t m_centers_init;
	// Maximum number of iterations to use when performing k-means clustering
	int m_iterations;
	// The data set used by this index
	DynamicMat& m_dataset;

	/** Attributes of the tree **/
	// The branching factor used in the hierarchical k-means clustering
	int m_branching;
	// Depth levels
	int m_depth;
	// Length of each feature.
	size_t m_veclen;
	// Number of nodes in the tree
	size_t m_size;
	// The root node in the tree.
	VocabTreeNodePtr m_root;

	/** Attributes of the database **/
	// Words of the vocabulary
	std::vector<VocabTreeNodePtr> m_words;
	// Number of DB images, used for creating the scores matrix
	uint m_numDbImages;

	/** Attributes holding the indices **/
	bfeat::DirectIndex* m_directIndex;

	/** Attributes used by several methods **/
	// The distance
	Distance m_distance;

public:

	/**
	 * Class constructor.
	 *
	 * @inputData - Reference to the matrix with the data to be clustered
	 * @param params - Parameters to the hierarchical k-means algorithm
	 */
	VocabTree(DynamicMat& inputData = DEFAULT_INPUTDATA,
			const cvflann::IndexParams& params = VocabTreeParams());

	/**
	 * Class destroyer, releases the memory used by the tree.
	 */
	virtual ~VocabTree();

	/**
	 * Returns the vocabulary size.
	 *
	 * @return number of leaf nodes in the tree
	 */
	size_t size();

	/**
	 * Builds the tree.
	 *
	 * @param inputData - Reference to a map with the name of the files containing the data to be clustered
	 *
	 * @note After this method is executed m_root holds a pointer to the tree,
	 *		 while m_words holds pointers to the leaf nodes.
	 * @note Interior nodes have only 'center' and 'children' information,
	 * 		 while leaf nodes have only 'center' and 'word_id', all weights for
	 * 		 interior nodes are 0 while weights for leaf nodes are 1.
	 */
	void build();

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
	 * Saves the inverted index to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the index
	 */
	void saveInvertedIndex(const std::string& filename) const;

	/**
	 * Loads the inverted index from a file stream.
	 *
	 * @param filename - The name of the file stream from where to load the index
	 */
	void loadInvertedIndex(const std::string& filename);

	/**
	 * Saves the direct index to a file stream.
	 *
	 * @param filename - The name of the file stream where to save the index
	 */
	void saveDirectIndex(const std::string& filename) const;

	/**
	 * Loads the direct index from a file stream.
	 *
	 * @param filename - The name of the file stream from where to load the index
	 */
	void loadDirectIndex(const std::string& filename);

	/**
	 * Quantizes a single feature vector into a word. Traverses the whole tree,
	 * and stores the resulting word id and weight.
	 *
	 * @param feature - Row vector representing the feature vector to quantize
	 * @param id - The id of the found word
	 * @param weight - The weight of the found word
	 * @param nodeAtL - Pointer to a tree node at level l in the path down the tree
	 */
	void quantize(const cv::Mat& feature, uint &id, double &weight,
			int &nodeAtL) const;

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
	void computeWordsWeights(WeightingType weighting);

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
	 * @param normType
	 */
	void normalizeDatabase(int normType = cv::NORM_L1);

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
	 * @param scores - Row matrix of size [1 x n] where n is the number DB images
	 * @param scoringMethod - normalization method used for scoring BoW vectors
	 *
	 * @note DB BoW vectors must be normalized beforehand
	 */
	void scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
			const int normType = cv::NORM_L2) const;

	/**
	 * Retrieves a DB BoW vector given its index.
	 *
	 * @param idx - The index of the DB image
	 * @param dbBowVector - A reference to the matrix where BoW vector will be save
	 */
	void getDbBoWVector(uint idx, cv::Mat& dbBowVector) const;

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

	int getWordId(size_t i) const {
		return m_words[i]->word_id;
	}

	double getWordWeight(size_t i) const {
		return m_words[i]->weight;
	}

	const std::vector<ImageCount>& getWordImageList(size_t i) const {
		return m_words[i]->image_list;
	}

	int getDirectIndexLevel() {
		return m_directIndex->getLevel();
	}

	void setDirectIndexLevel(int levelsUp) {
		int directIndexLevel = m_depth - 1 - levelsUp;
		if (directIndexLevel < 0) {
			directIndexLevel = 0;
		} else if (directIndexLevel > m_depth - 1) {
			directIndexLevel = m_depth - 1;
		}
		m_directIndex->setLevel(directIndexLevel);
	}

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
	void load_tree(cv::FileNode& fs, VocabTreeNodePtr& node);

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

	bool compareEqual(const VocabTreeNodePtr a, const VocabTreeNodePtr b) const;

	// Make private the copy constructor and the assignment operator
	// to prevent obtaining copies of the instance
	VocabTree(VocabTree const&); // Don't Implement
	void operator=(VocabTree const&); // Don't implement

};

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
VocabTree<TDescriptor, Distance>::VocabTree(DynamicMat& inputData,
		const cvflann::IndexParams& params) :
		m_dataset(inputData), m_veclen(0), m_size(0), m_root(NULL), m_distance(
				Distance()) {

	// Attributes initialization
	m_veclen = m_dataset.cols;
	m_branching = get_param(params, "branching", 6);
	m_iterations = get_param(params, "iterations", 1);
	m_depth = get_param(params, "depth", 10);
	m_centers_init = get_param(params, "centers_init",
			cvflann::FLANN_CENTERS_RANDOM);
	m_directIndex = new bfeat::DirectIndex();
	setDirectIndexLevel(get_param(params, "levels_up", 0));
	m_numDbImages = 0;

	if (m_iterations < 0) {
		m_iterations = (std::numeric_limits<int>::max)();
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
VocabTree<TDescriptor, Distance>::~VocabTree() {
	delete m_directIndex;
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
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
size_t VocabTree<TDescriptor, Distance>::size() {
	return m_words.size();
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
	size_t size = m_dataset.rows;

	//  Array of descriptors indices
	int* indices = new int[size];
	for (size_t i = 0; i < size; ++i) {
		indices[i] = int(i);
	}

	m_root = new VocabTreeNode();
	computeNodeStatistics(m_root, indices, (int) size);

#if VTREEVERBOSE
	printf("[VocabTree::build] Started clustering\n");
#endif

	computeClustering(m_root, indices, (int) size, 0, false);

#if VTREEVERBOSE
	printf("[VocabTree::build] Finished clustering\n");
#endif

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::save(const std::string& filename) const {

	if (empty()) {
		throw std::runtime_error(
				"[VocabTree::save] Error while saving tree, vocabulary is empty");
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
	fs << "numDbImages" << (int) m_numDbImages;

	fs << "root";

	save_tree(fs, m_root);

	fs.release();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::save_tree(cv::FileStorage& fs,
		VocabTreeNodePtr node) const {

	// WriteNode
	fs << "{";
	fs << "center"
			<< cv::Mat(1, m_veclen, cv::DataType<TDescriptor>::type,
					(uchar*) node->center);
	fs << "nodeId" << node->node_id;
	fs << "wordId" << node->word_id;

	fs << "children" << "[";
	if (node->children != NULL) {
		// WriteChildren
		for (size_t i = 0; (int) i < m_branching; i++) {
			save_tree(fs, node->children[i]);
		}
	}
	fs << "]";

	fs << "}";
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::load(const std::string& filename) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("Could not open file " + filename);
	}

	m_iterations = (int) fs["iterations"];
	m_branching = (int) fs["branching"];
	m_depth = (int) fs["depth"];
	m_veclen = (int) fs["vectorLength"];
	m_numDbImages = (int) fs["numDbImages"];

	cv::FileNode root = fs["root"];

	m_root = new VocabTreeNode();

	load_tree(root, m_root);

	fs.release();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::load_tree(cv::FileNode& fs,
		VocabTreeNodePtr& node) {

	cv::Mat center;
	fs["center"] >> center;
	CV_Assert(center.rows == 1);
	CV_Assert(center.cols == (int ) m_veclen);

	// Deep copy
	node->center = new TDescriptor[m_veclen];
	for (size_t k = 0; k < m_veclen; ++k) {
		node->center[k] = center.at<TDescriptor>(0, k);
	}
	center.release();

	node->node_id = (int) fs["nodeId"];
	node->word_id = (int) fs["wordId"];

	cv::FileNode children = fs["children"];

	if (children.size() == 0) {
		// Node has no children then it's a leaf node

		node->children = NULL;

		std::vector<ImageCount>().swap(node->image_list);

		m_words.push_back(node);

	} else {
		// Node has children then it's an interior node

		// Verifying that children is a sequence
		if (children.type() != cv::FileNode::NONE
				&& children.type() != cv::FileNode::SEQ) {
			throw std::runtime_error("Error while parsing tree,"
					" fetched element 'children' should be a sequence");
		}

		// Verifying that children has 0 or k elements
		if (children.size() != 0 && (int) children.size() != m_branching) {
			std::stringstream ss;
			ss << "Error while parsing tree, fetched element"
					" 'children' must have [0] or [" << m_branching
					<< "] elements";
			throw std::runtime_error(ss.str());
		}

		node->children = new VocabTreeNodePtr[m_branching];
		cv::FileNodeIterator it = children.begin();

		for (size_t c = 0; (int) c < m_branching; ++c, ++it) {
			node->children[c] = new VocabTreeNode();
			cv::FileNode child = *it;
			load_tree(child, node->children[c]);
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::saveInvertedIndex(
		const std::string& filename) const {

	if (m_words.empty() == true) {
		throw std::runtime_error("[VocabTree::saveInvertedIndex] "
				"Vocabulary is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[VocabTree::saveInvertedIndex] "
				"Unable to open file "
				"[" + filename + "] for writing");
	}

	fs << "NumDBImages" << int(m_numDbImages);

	fs << "Words" << "[";
	for (VocabTreeNodePtr node : m_words) {
		fs << "{";

		fs << "weight" << node->weight;
		fs << "imageList" << "[";
		for (ImageCount img : node->image_list) {
			fs << "{:" << "m_index" << int(img.m_index) << "m_count"
					<< img.m_count << "}";
		}
		fs << "]";

		fs << "}";
	}

	fs << "]";

	fs.release();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::loadInvertedIndex(
		const std::string& filename) {

	if (m_words.empty() == true) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Vocabulary is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (fs.isOpened() == false) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Unable to open file [" + filename + "] for reading");
	}

	m_numDbImages = int(fs["NumDBImages"]);

	cv::FileNode words = fs["Words"];

	// Verify that 'Words' is a sequence
	if (words.type() != cv::FileNode::SEQ) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Fetched element 'Words' should be a sequence");
	}

	if (words.size() != m_words.size()) {
		throw std::runtime_error("[VocabTree::loadInvertedIndex] "
				"Vocabulary size is different than inverted file length");
	}

	int wordId = 0;

	cv::FileNode images;

	for (cv::FileNodeIterator word = words.begin(); word != words.end();
			++word, ++wordId) {

		m_words[wordId]->weight = double((*word)["weight"]);

		images = (*word)["imageList"];

		// Verify that imageList is a sequence
		if (images.type() != cv::FileNode::SEQ) {
			throw std::runtime_error("[VocabTree::loadInvertedIndex] "
					"Fetched element 'imageList' is not a sequence");
		}

		for (cv::FileNodeIterator image = images.begin(); image != images.end();
				++image) {
			size_t index = int((*image)["m_index"]);
			ImageCount img(index, float((*image)["m_count"]));
			m_words[wordId]->image_list.push_back(img);
		}
	}

	fs.release();

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::saveDirectIndex(
		const std::string& filename) const {

	m_directIndex->save(filename);

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::loadDirectIndex(
		const std::string& filename) {

	m_directIndex->clear();

	m_directIndex->load(filename);

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::computeNodeStatistics(
		VocabTreeNodePtr node, int* indices, int indices_length) {

#if VTREEVERBOSE
	printf("[VocabTree::build] Computing root node statistics\n");
#endif

	TDescriptor* center = new TDescriptor[m_veclen];

	// Checking indices to be in range, but actually is not necessary
	// is just for avoing the unused warning
	for (size_t i = 0; (int) i < indices_length; ++i) {
		CV_Assert(0 <= indices[i] && indices[i] < m_dataset.rows);
	}

	// Zeroing all centroid dimensions
	cv::Mat centroid = cv::Mat::zeros(1, m_veclen, m_dataset.type());

	// Setting the centroid
	for (size_t k = 0; k < m_veclen; ++k) {
		center[k] = centroid.at<TDescriptor>(0, k);
	}
	centroid.release();

	node->center = center;
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
	// or when there are less data than clusters
	if (level == m_depth - 1 || indices_length < m_branching) {
		node->children = NULL;
		node->word_id = m_words.size();
		node->weight = 1.0;
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
	if (centers_length < m_branching) {
		node->children = NULL;
		node->word_id = m_words.size();
		node->weight = 1.0;
		m_words.push_back(node);
#if VTREEVERBOSE
		printf(
				"[VocabTree::computeClustering] (level %d): got less cluster indices than clusters (%d features)\n",
				level, indices_length);
#endif
		return;
	}

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
	while (!converged && iteration < m_iterations) {
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
				KMajorityIndex::cumBitSum(m_dataset.row(indices[i]), b);
			}
			// Bitwise majority voting
			for (size_t j = 0; (int) j < m_branching; ++j) {
				cv::Mat centroid = dcenters.row(j);
				KMajorityIndex::majorityVoting(bitwiseCount.row(j), centroid,
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

		// Handle empty clusters
		for (int i = 0; i < m_branching; ++i) {
			// if one cluster converges to an empty cluster,
			// move an element into that cluster
			if (count[i] == 0) {
				int j = (i + 1) % m_branching;
				while (count[j] <= 1) {
					j = (j + 1) % m_branching;
				}

				for (int k = 0; k < indices_length; ++k) {
					if (belongs_to[k] == j) {
						belongs_to[k] = i;
						--count[j];
						++count[i];
						break;
					}
				}
				converged = false;
			}
		}

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

		node->children[c] = new VocabTreeNode();
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
void VocabTree<TDescriptor, Distance>::transform(const cv::Mat& featuresVector,
		cv::Mat& bowVector, const int& normType) const {

	// Initialize query BoW vector
	bowVector = cv::Mat::zeros(1, m_words.size(), cv::DataType<float>::type);

	uint wordIdx;
	double wordWeight;
	int nodeAtL; // unused

	// Quantize each query image feature vector
	for (size_t i = 0; (int) i < featuresVector.rows; ++i) {

		quantize(featuresVector.row(i), wordIdx, wordWeight, nodeAtL);

		if (wordIdx > m_words.size() - 1) {
			throw std::runtime_error(
					"[VocabTree::transform] Feature quantized into a non-existent word");
		}

		bowVector.at<float>(0, wordIdx) += (float) wordWeight;
	}

	//	Normalizing query BoW vector
	cv::normalize(bowVector, bowVector, 1, 0, normType);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::quantize(const cv::Mat& feature,
		uint &word_id, double &weight, int &nodeAtL) const {

	VocabTreeNodePtr best_node = m_root;

	int level = 0;

	while (best_node->children != NULL) {

		VocabTreeNodePtr node = best_node;

		// Arbitrarily assign to first child
		best_node = node->children[0];
		DistanceType best_distance = m_distance((TDescriptor*) feature.data,
				best_node->center, m_veclen);

		// Looking for a better child
		for (size_t j = 1; (int) j < m_branching; ++j) {
			DistanceType d = m_distance((TDescriptor*) feature.data,
					node->children[j]->center, m_veclen);
			if (d < best_distance) {
				best_distance = d;
				best_node = node->children[j];
				if (level == m_directIndex->getLevel()) {
					nodeAtL = j;
				}
			}
		}
		++level;
	}

	// Turn node id into word id
	word_id = best_node->word_id;
	weight = best_node->weight;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::computeWordsWeights(
		WeightingType weighting) {

	if (empty()) {
		throw std::runtime_error("[VocabTree::computeWordsWeights]"
				" Error while computing words weights, vocabulary is empty");
	}

	if (weighting == bfeat::BINARY) {
		// Setting constant weight equal to 1
		for (VocabTreeNodePtr& word : m_words) {
			word->weight = 1.0;
		}
	} else if (weighting == bfeat::TF_IDF) {
		// Calculating the IDF part of the TF-IDF score, the complete
		// TF-IDF score is the result of multiplying the weight by the word count
		for (VocabTreeNodePtr& word : m_words) {
			int len = word->image_list.size();
			// because having that a descriptor from all DB images is quantized
			// to the same word is quite unlikely
			if (len > 0) {
				word->weight = log((double) m_numDbImages / (double) len);
			} else {
				word->weight = 0.0;
			}
		}
	} else {
		throw std::runtime_error(
				"[VocabTree::computeWordsWeights] Unknown weighting type");
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::createDatabase() {

	if (empty()) {
		throw std::runtime_error("[VocabTree::createDatabase] Error while"
				" applying weights to words histogram, vocabulary is empty");
	}

	// Loop over words
	for (VocabTreeNodePtr& word : m_words) {
		// Apply word weight to the image count
		for (ImageCount& image : word->image_list) {
			image.m_count *= word->weight;
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::clearDatabase() {
	for (VocabTreeNodePtr& word : m_words) {
		std::vector<ImageCount>().swap(word->image_list);
	}
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
bool VocabTree<TDescriptor, Distance>::empty() const {
	return m_words.empty();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::addImageToDatabase(uint imgIdx,
		cv::Mat imgFeatures) {

	if (imgFeatures.rows < 1) {
		throw std::runtime_error(
				"[VocabTree::addImageToDatabase] Error while adding image, at least one feature vector is needed");
	}

	if (imgFeatures.cols != (int) m_veclen) {
		std::stringstream ss;
		ss << "Error while adding image, feature vector has different length"
				" than the ones used for building the tree, it is ["
				<< imgFeatures.cols << "] while it should be[" << m_veclen
				<< "]";
		throw std::runtime_error(ss.str());
	}

	if (m_words.empty() == true) {
		throw std::runtime_error(
				"[VocabTree::addImageToDatabase] Error while adding image,"
						" vocabulary is empty");
	}

	uint wordIdx;
	double wordWeight; // not needed
	int nodeAtL;

	for (size_t i = 0; (int) i < imgFeatures.rows; ++i) {
		quantize(imgFeatures.row(i), wordIdx, wordWeight, nodeAtL);
		addFeatureToInvertedFile(wordIdx, imgIdx);
		m_directIndex->addFeature(imgIdx, nodeAtL, i);
	}

	// Increasing the counter of images in the DB
	++m_numDbImages;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::addFeatureToInvertedFile(uint wordIdx,
		uint imgIdx) {

	int n = (int) m_words[wordIdx]->image_list.size();

	// Images list is empty: push a new image
	if (n == 0) {
		m_words[wordIdx]->image_list.push_back(ImageCount(imgIdx, (float) 1.0));
	} else {
		// Images list is not empty: check if the id of the last added image
		// is the same than that of the image being added
		if (m_words[wordIdx]->image_list[n - 1].m_index == imgIdx) {
			// Images are equal then the counter is increased by one
			m_words[wordIdx]->image_list[n - 1].m_count += (float) 1.0;
		} else {
			// Images are different then push a new image
			m_words[wordIdx]->image_list.push_back(
					ImageCount(imgIdx, (float) 1.0));
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::normalizeDatabase(int normType) {

	if (empty()) {
		throw std::runtime_error("[VocabTree::normalizeDatabase] Error while"
				" normalizing DB BoW vectors, vocabulary is empty");
	}

	// Magnitude of a vector is defined as: sum(abs(xi)^p)^(1/p)

	std::vector<float> mags(m_numDbImages, 0.0);

	// Computing DB BoW vectors magnitude

	// Summing vector elements
	for (VocabTreeNodePtr& word : m_words) {
		for (ImageCount& image : word->image_list) {
			uint index = image.m_index;
			double dim = image.m_count;

			assert(index < mags.size());

			if (normType == cv::NORM_L1) {
				mags[index] += fabs(dim);
			} else if (normType == cv::NORM_L2) {
				mags[index] += pow(dim, 2);
			} else {
				throw std::runtime_error(
						"[VocabTree::normalizeDatabase] Unknown scoring method");
			}
		}
	}

	// Applying power over sum result
	if (normType == cv::NORM_L2) {
		for (size_t i = 0; i < mags.size(); ++i) {
			mags[i] = sqrt(mags[i]);
		}
	}

	// Normalizing database
	for (VocabTreeNodePtr& word : m_words) {
		for (ImageCount& image : word->image_list) {
			uint index = image.m_index;
			assert(index < mags.size());
			if (mags[index] > 0.0) {
				image.m_count /= mags[index];
			}
		}
	}

}

// --------------------------------------------------------------------------

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::scoreQuery(
		const cv::Mat& queryImgFeatures, cv::Mat& scores,
		const int normType) const {

	if (queryImgFeatures.rows < 1) {
		throw std::runtime_error(
				"[VocabTree::scoreQuery] Error while scoring image, at least one feature vector is needed");
	}

	if (queryImgFeatures.cols != (int) m_veclen) {
		std::stringstream ss;
		ss << "Error while adding image, feature vector has different length"
				" than the ones used for building the tree, it is ["
				<< queryImgFeatures.cols << "] while it should be[" << m_veclen
				<< "]";
		throw std::runtime_error(ss.str());
	}

	if (m_words.empty() == true) {
		throw std::runtime_error("[VocabTree::scoreQuery]"
				" Error while scoring query, vocabulary is empty");
	}

	if (normType != cv::NORM_L1 && normType != cv::NORM_L2) {
		throw std::runtime_error(
				"[VocabTree::scoreQuery] Unknown scoring method");
	}

	scores = cv::Mat::zeros(1, m_numDbImages, cv::DataType<float>::type);

	cv::Mat queryBowVector;
	transform(queryImgFeatures, queryBowVector, normType);

	//	Efficient scoring query BoW vector against all DB BoW vectors

	// ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
	// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )

	// Calculating sum part of the efficient score implementation
	for (VocabTreeNodePtr word : m_words) {
		float qi = queryBowVector.at<float>(0, word->word_id);

		// Early exit
		if (qi == 0.0) {
			continue;
		}

		// The inverted file of a word contains all images counts quantized into that word
		// i.e. if they are there its because their count di is not zero

		// In addition its fair computing qi against di without further verification
		// since the inverted files contain not null counts

		for (ImageCount& image : word->image_list) {
			float di = image.m_count;

			// qi cannot be zero because we are considering only when its non-zero
			// qi cannot be more than 1 because it is supposed to be normalized
			CV_Assert(qi > 0 && qi <= 1.0);

			// di cannot more than 1 because it is supposed to be normalized
			CV_Assert(di <= 1.0);

			// di cannot be zero (unless the weight is zero) because the inverted files
			// contain only counts for images with a descriptor which was quantized
			// into that word
			if (word->weight != 0.0) {
				CV_Assert(di > 0.0);
			} else {
				CV_Assert(di >= 0.0);
			}

			if (normType == cv::NORM_L1) {
				scores.at<float>(0, image.m_index) += (float) (fabs(qi - di)
						- fabs(qi) - fabs(di));
			} else if (normType == cv::NORM_L2) {
				scores.at<float>(0, image.m_index) += (float) qi * di;
			}
		}
	}

	// Completing efficient score implementation
	for (int i = 0; i < scores.cols; ++i) {
		if (normType == cv::NORM_L1) {
			scores.at<float>(0, i) = (float) (-scores.at<float>(0, i) / 2.0);
		} else if (normType == cv::NORM_L2) {
			scores.at<float>(0, i) =
					(float) (2.0 - 2.0 * scores.at<float>(0, i));
		}
		// else, not possible since normType was validated before
	}

}

template<class TDescriptor, class Distance>
void VocabTree<TDescriptor, Distance>::getDbBoWVector(uint idx,
		cv::Mat& dbBowVector) const {

	if (m_words.empty() == true) {
		throw std::runtime_error(
				"[VocabTree::getDbBoWVector] Error while obtaining DB BoW vectors,"
						" vocabulary is empty");
	}

	dbBowVector = cv::Mat::zeros(1, m_words.size(), cv::DataType<float>::type);

	for (VocabTreeNodePtr word : m_words) {
		for (ImageCount& image : word->image_list) {
			if (image.m_index == idx) {
				dbBowVector.at<float>(0, word->word_id) = image.m_count;
			}
		}
	}
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

} /* namespace cvflann */

#endif /* VOCAB_TREE_H_ */
