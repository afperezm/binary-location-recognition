/*
 * HCTree.cpp
 *
 *  Created on: Jan 20, 2014
 *      Author: andresf
 */

#include <HCTree.hpp>

#include <fstream>

namespace vlr {

HCTree::HCTree(vlr::Mat& inputData, const HCTreeParams& params) :
		m_dataset(inputData), m_veclen(0), m_size(0), m_root(NULL), m_distance(
				Distance()) {

	// Attributes initialization
	m_branching = cvflann::get_param(params, "branching", 16);
	m_maxLeafSize = cvflann::get_param(params, "maxLeafSize", 150);
	m_veclen = m_dataset.cols;

}

// --------------------------------------------------------------------------

HCTree::~HCTree() {
	if (m_root != NULL) {
		free_centers(m_root);
	}
}

// --------------------------------------------------------------------------

void HCTree::free_centers(HCTreeNodePtr node) {
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

size_t HCTree::size() const {
	return m_size;
}

// --------------------------------------------------------------------------

bool HCTree::empty() const {
	return m_size == 0;
}

// --------------------------------------------------------------------------

bool HCTree::operator==(const HCTree &other) const {

	if (m_branching != other.getBranching()
			|| m_maxLeafSize != other.getMaxLeafSize()
			|| m_veclen != other.getVeclen() || m_size != other.size()) {
		return false;
	}

	if (compareEqual(m_root, other.getRoot()) == false) {
		return false;
	}

	return true;
}

bool HCTree::compareEqual(const HCTreeNodePtr a, const HCTreeNodePtr b) const {

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
		for (int i = 0; i < m_branching; ++i) {
			if (compareEqual(a->children[i], b->children[i]) == false) {
				return false;
			}
		}
	}

	return true;
}
// --------------------------------------------------------------------------

bool HCTree::operator!=(const HCTree &other) const {
	return !(*this == other);
}

// --------------------------------------------------------------------------

void HCTree::save(const std::string& filename) const {

	if (empty()) {
		throw std::runtime_error("[HCTree::save] Tree is empty");
	}

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (fs.isOpened() == false) {
		throw std::runtime_error(
				"[HCTree::save] Error opening file [" + filename
						+ "] for writing");
	}

	fs << "branching" << m_branching;
	fs << "maxLeafSize" << m_maxLeafSize;
	fs << "vectorLength" << (int) m_veclen;
	fs << "size" << (int) m_size;

	fs << "nodes" << "[";

	save_tree(fs, m_root);

	fs << "]";

	fs.release();
}

// --------------------------------------------------------------------------

void HCTree::save_tree(cv::FileStorage& fs, HCTreeNodePtr node) const {

	// Save node information
	fs << "{";
	fs << "nodeId" << node->nodeId;
	fs << "center"
			<< cv::Mat(1, m_veclen, cv::DataType<TDescriptor>::type,
					(uchar*) node->center);
	fs << "hasChildren" << int(node->children != NULL);
	fs << "}";

	// Save children, if any
	if (node->children != NULL) {
		for (int i = 0; i < m_branching; ++i) {
			save_tree(fs, node->children[i]);
		}
	}
}

// --------------------------------------------------------------------------

void HCTree::load(const std::string& filename) {

	std::ifstream inputZippedFileStream;
	boost::iostreams::filtering_istream inputFileStream;

	std::string line, field;
	std::stringstream ss;

	enum treeFields {
		branching, maxLeafSize, vectorLength, size, nodes
	};
	std::string treeFieldsNames[] = { "branching:", "maxLeafSize:",
			"vectorLength:", "size:", "nodes:" };

	// Open file
	inputZippedFileStream.open(filename.c_str(),
			std::fstream::in | std::fstream::binary);

	// Check file
	if (inputZippedFileStream.good() == false) {
		throw std::runtime_error("[HCTree::load] "
				"Unable to open file [" + filename + "] for reading");
	}

	try {
		inputFileStream.push(boost::iostreams::gzip_decompressor());
		inputFileStream.push(inputZippedFileStream);

		while (getline(inputFileStream, line)) {
			ss.clear();
			ss.str(line);
			ss >> field;
			if (field.compare(treeFieldsNames[branching]) == 0) {
				ss >> m_branching;
			} else if (field.compare(treeFieldsNames[maxLeafSize]) == 0) {
				ss >> m_maxLeafSize;
			} else if (field.compare(treeFieldsNames[vectorLength]) == 0) {
				ss >> m_veclen;
			} else if (field.compare(treeFieldsNames[size]) == 0) {
				ss >> m_size;
			} else if (field.compare(treeFieldsNames[nodes]) == 0) {
				break;
			}
		}

		m_root = new HCTreeNode();
		load_tree(inputFileStream, m_root);

	} catch (const boost::iostreams::gzip_error& e) {
		throw std::runtime_error("[HCTree::load] "
				"Got error while parsing file [" + std::string(e.what()) + "]");
	}

	// Close file
	inputZippedFileStream.close();

}

// --------------------------------------------------------------------------

void HCTree::load_tree(boost::iostreams::filtering_istream& inputFileStream,
		HCTreeNodePtr& node) {

	enum nodeFields {
		start, nodeId, center, rows, cols, dt, data, hasChildren
	};
	std::string nodeFieldsNames[] = { "-", "nodeId:", "center:", "rows:",
			"cols:", "dt:", "data:", "hasChildren:" };

	std::string line, field;
	std::stringstream ss;

	cv::Mat _center;

	int _rows = -1;
	int _cols = -1;
	std::string _type;
	int colIdx = -1;
	float elem;
	bool _hasChildren;

	while (getline(inputFileStream, line)) {
		ss.clear();
		ss.str(line);
		ss >> field;
		if (field.compare(nodeFieldsNames[start]) == 0) {
			continue;
		} else if (field.compare(nodeFieldsNames[nodeId]) == 0) {
			ss >> node->nodeId;
		} else if (field.compare(nodeFieldsNames[center]) == 0) {
			continue;
		} else if (field.compare(nodeFieldsNames[rows]) == 0) {
			ss >> _rows;
		} else if (field.compare(nodeFieldsNames[cols]) == 0) {
			ss >> _cols;
		} else if (field.compare(nodeFieldsNames[dt]) == 0) {
			ss >> _type;
		} else if (field.compare(nodeFieldsNames[hasChildren]) == 0) {
			ss >> _hasChildren;
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

	if (_hasChildren == false) {
		// Node has no children then it's a leaf node
		node->children = NULL;
	} else {
		// Node has children then it's an interior node
		node->children = new HCTreeNodePtr[m_branching];
		for (int c = 0; c < m_branching; ++c) {
			node->children[c] = new HCTreeNode();
			load_tree(inputFileStream, node->children[c]);
		}
	}

}

// --------------------------------------------------------------------------

void HCTree::build() {

	if (m_branching < 2) {
		throw std::runtime_error("[HCTree::build] Error, branching factor"
				" must be at least 2");
	}

	if (m_dataset.empty()) {
		throw std::runtime_error("[HCTree::build] Error, data set is empty"
				" cannot proceed with clustering");
	}

	// Number of features in the data set
	int size = m_dataset.rows;

	//  Array of descriptors indices
	int* indices = new int[size];
	for (int i = 0; i < size; ++i) {
		indices[i] = i;
	}

	m_root = new HCTreeNode();

	m_root->center = new TDescriptor[m_veclen];
	std::fill(m_root->center, m_root->center + m_veclen, 0);

#if HCTREEVERBOSE
	printf("[HCTree::build] Started clustering\n");
#endif

	computeClustering(m_root, indices, size, 0, false);

#if HCTREEVERBOSE
	printf("[HCTree::build] Finished clustering\n");
#endif

	delete[] indices;
}

// --------------------------------------------------------------------------

void HCTree::computeClustering(HCTreeNodePtr node, int* indices,
		int indices_length, int level, bool fitted) {

	// Assign node id then increase nodes counter
	node->nodeId = m_size++;

	// Sort descriptors, caching leverages this fact
	// Note: it doesn't affect the clustering process since all descriptors referenced by indices belong to the same cluster
	if (level > 0) {
		std::sort(indices, indices + indices_length);
	}

	// Recursion base case: done when the minimum leaf size was reached
	// or when there is less data than clusters
	if (indices_length < m_maxLeafSize || indices_length < m_branching) {
		node->children = NULL;
#if HCTREEVERBOSE
		printf("[HCTree::computeClustering] (level %d): reached minimum leaf size "
				"or got less data than clusters (%d features)\n", level, indices_length);
#endif
		return;
	}

#if HCTREEVERBOSE
	printf("[HCTree::computeClustering] (level %d): running k-means (%d features)\n",
			level, indices_length);
#endif

	std::vector<int> centers_idx(m_branching);
	int centers_length;

#if DEBUG
#if HCTREEVERBOSE
	printf("randomCenters - Start\n");
#endif
#endif

	CentersChooser<TDescriptor, Distance>::create(cvflann::FLANN_CENTERS_RANDOM)->chooseCenters(
			m_branching, indices, indices_length, centers_idx, centers_length,
			m_dataset);

#if DEBUG
#if HCTREEVERBOSE
	printf("randomCenters - End\n");
#endif
#endif

	// Recursion base case: done as well if by case got
	// less cluster indices than clusters
#ifdef SUPPDUPLICATES
	if (centers_length < m_branching) {
		node->children = NULL;
#if HCTREEVERBOSE
		printf("[HCTree::computeClustering] (level %d): got less cluster indices than clusters (%d features)\n",
				level, indices_length);
#endif
		return;
	}
#else
	CV_Assert(centers_length == m_branching);
#endif

#if DEBUG
#if HCTREEVERBOSE
	printf("initCentroids - Start\n");
#endif
#endif

	cv::Mat dcenters(m_branching, m_veclen, m_dataset.type());
	for (int i = 0; i < centers_length; ++i) {
		m_dataset.row(centers_idx[i]).copyTo(
				dcenters(cv::Range(i, i + 1), cv::Range(0, m_veclen)));
	}

#if DEBUG
#if HCTREEVERBOSE
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
#if HCTREEVERBOSE
		printf("Clearing cache at level=[%d]\n", level);
#endif
	}

#if DEBUG
#if HCTREEVERBOSE
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
#if HCTREEVERBOSE
	printf("quantize - End\n");
#endif
#endif

	TDescriptor** centers = new TDescriptor*[m_branching];

	for (int i = 0; i < m_branching; ++i) {
		centers[i] = new TDescriptor[m_veclen];
		for (size_t k = 0; k < m_veclen; ++k) {
			centers[i][k] = dcenters.at<TDescriptor>(i, k);
		}
	}

	// Compute clustering for each of the resulting clusters
	node->children = new HCTreeNodePtr[m_branching];
	int start = 0;
	int end = start;
	for (int c = 0; c < m_branching; ++c) {

#if HCTREEVERBOSE
		printf("[HCTree::computeClustering] Clustering over resulting clusters, "
				"level=[%d] branch=[%d]\n", level, c);
#endif

		// Re-order indices by chunks in clustering order
		for (int i = 0; i < indices_length; ++i) {
			if (belongs_to[i] == c) {
				std::swap(indices[i], indices[end]);
				std::swap(belongs_to[i], belongs_to[end]);
				++end;
			}
		}

		node->children[c] = new HCTreeNode();
		node->children[c]->center = centers[c];
		computeClustering(node->children[c], indices + start, end - start,
				level + 1, fitted);
		start = end;
	}

	dcenters.release();
	delete[] centers;

}

} /* namespace vlr */
