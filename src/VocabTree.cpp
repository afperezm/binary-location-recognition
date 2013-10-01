/*
 * VocabTree.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: andresf
 */

#include <VocabTree.h>
#include <KMajorityIndex.h>

namespace cvflann {

// --------------------------------------------------------------------------
VocabTree::VocabTree(const cv::Mat& inputData, const IndexParams& params) :
		m_dataset(inputData), m_veclen(0), m_root(NULL), m_distance(Distance()), m_memoryCounter(
				0) {

	// Attributes initialization
	m_veclen = m_dataset.cols;
	m_branching = get_param(params, "branching", 6);
	m_iterations = get_param(params, "iterations", 11);
	m_depth = get_param(params, "depth", 10);
	m_centers_init = get_param(params, "centers_init", FLANN_CENTERS_RANDOM);

	if (m_iterations < 0) {
		m_iterations = (std::numeric_limits<int>::max)();
	}

	m_words.clear();
}

// --------------------------------------------------------------------------

VocabTree::~VocabTree() {
	if (m_root != NULL) {
		free_centers(m_root);
	}
}

// --------------------------------------------------------------------------

void VocabTree::build() {

	if (m_branching < 2) {
		throw std::runtime_error("Branching factor must be at least 2");
	}

	// Number of features in the dataset
	size_t size = m_dataset.rows;

	//  Array of indices to vectors in the dataset
	int* indices = new int[size];
	for (size_t i = 0; i < size; ++i) {
		indices[i] = int(i);
	}

	printf("[BHCIndex::buildIndex] Building tree from %d features\n",
			(int) size);
	printf("[BHCIndex::buildIndex]   with depth %d, branching factor %d\n",
			m_depth, m_branching);
	printf("[BHCIndex::buildIndex]   and restarts %d\n", m_iterations);

	m_root = m_pool.allocate<VocabTreeNode>();
	computeNodeStatistics(m_root, indices, (int) size);
	computeClustering(m_root, indices, (int) size, 0);
}

// --------------------------------------------------------------------------

void VocabTree::save(FILE* stream) const {
	save_value(stream, m_centers_init);
	save_value(stream, m_iterations);
	save_value(stream, m_branching);
	save_value(stream, m_depth);
	save_value(stream, m_veclen);
	save_value(stream, m_distance);
	save_value(stream, m_memoryCounter);

	save_tree(stream, m_root);
}

// --------------------------------------------------------------------------

void VocabTree::load(FILE* stream) {
	load_value(stream, m_centers_init);
	load_value(stream, m_iterations);
	load_value(stream, m_branching);
	load_value(stream, m_depth);
	load_value(stream, m_veclen);
	load_value(stream, m_distance);
	load_value(stream, m_memoryCounter);

	if (m_root != NULL) {
		free_centers(m_root);
	}
	load_tree(stream, m_root);
}

// --------------------------------------------------------------------------

int VocabTree::usedMemory() const {
	return m_pool.usedMemory + m_pool.wastedMemory + m_memoryCounter;
}

// --------------------------------------------------------------------------

void VocabTree::save_tree(FILE* stream, VocabTreeNodePtr node) const {
	save_value(stream, *node);
//	save_value(stream, *(node->center), (int) m_veclen);
	if (node->children == NULL) {
//		int indices_offset = (int) (node->indices - m_indices);
//		save_value(stream, indices_offset);
	} else {
		for (int i = 0; i < m_branching; ++i) {
			save_tree(stream, node->children[i]);
		}
	}
}

// --------------------------------------------------------------------------

void VocabTree::load_tree(FILE* stream, VocabTreeNodePtr& node) {
	node = m_pool.allocate<VocabTreeNode>();
	load_value(stream, *node);
//	node->center = new TDescriptor[m_veclen];
//	load_value(stream, *(node->center), (int) m_veclen);
	if (node->children == NULL) {
		int indices_offset;
		load_value(stream, indices_offset);
//		node->indices = m_indices + indices_offset;
	} else {
		node->children = m_pool.allocate<VocabTreeNodePtr>(m_branching);
		for (int i = 0; i < m_branching; ++i) {
			load_tree(stream, node->children[i]);
		}
	}
}

// --------------------------------------------------------------------------

void VocabTree::free_centers(VocabTreeNodePtr node) {
	delete[] node->center;
	if (node->children != NULL) {
		for (int k = 0; k < m_branching; ++k) {
			free_centers(node->children[k]);
		}
	}
}

// --------------------------------------------------------------------------

void VocabTree::computeNodeStatistics(VocabTreeNodePtr node, int* indices,
		int indices_length) {

	TDescriptor* center = new TDescriptor[m_veclen];

	m_memoryCounter += int(m_veclen * sizeof(TDescriptor));

	// Compute center using majority voting over all data
	cv::Mat accVector(1, m_veclen * 8, cv::DataType<int>::type);
	accVector = cv::Scalar::all(0);
	for (size_t i = 0; (int) i < indices_length; i++) {
		KMajorityIndex::cumBitSum(m_dataset.row(indices[i]), accVector);
	}
	cv::Mat centroid(1, m_veclen, m_dataset.type());
	KMajorityIndex::majorityVoting(accVector, centroid, indices_length);

	for (size_t k = 0; k < m_veclen; ++k) {
		center[k] = centroid.at<TDescriptor>(0, k);
	}

	node->center = center;
}

// --------------------------------------------------------------------------

void VocabTree::computeClustering(VocabTreeNodePtr node, int* indices,
		int indices_length, int level) {
//	node->size = indices_length;

	// Recursion base case: done when the last level is reached
	// or when there are less data than clusters
	if (level == m_depth - 1 || indices_length < m_branching) {
//		node->indices = indices;
//		std::sort(node->indices, node->indices + indices_length);
		node->children = NULL;
		node->word_id = m_words.size();
		this->m_words.push_back(node);
		return;
	}

	printf("[BuildRecurse] (level %d): Running k-means (%d features)\n", level,
			indices_length);

	int* centers_idx = new int[m_branching];
	int centers_length;

	CentersChooser<Distance>::create(m_centers_init)->chooseCenters(m_branching,
			indices, indices_length, centers_idx, centers_length, m_dataset);

	// Recursion base case: done as well if by case got
	// less cluster indices than clusters
	if (centers_length < m_branching) {
//		node->indices = indices;
//		std::sort(node->indices, node->indices + indices_length);
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

	cv::Mat dcenters(m_branching, m_veclen, m_dataset.type());
	for (int i = 0; i < centers_length; i++) {
		m_dataset.row(centers_idx[i]).copyTo(
				dcenters(cv::Range(i, i + 1), cv::Range(0, m_veclen)));
	}
	delete[] centers_idx;

	std::vector<DistanceType> radiuses(m_branching);
	int* count = new int[m_branching];
	for (int i = 0; i < m_branching; ++i) {
		radiuses[i] = 0;
		count[i] = 0;
	}

	//TODO quantize: assign points to clusters
	int* belongs_to = new int[indices_length];
	for (int i = 0; i < indices_length; ++i) {

		DistanceType sq_dist = m_distance(m_dataset.row(indices[i]).data,
				dcenters.row(0).data, m_veclen);
		belongs_to[i] = 0;
		for (int j = 1; j < m_branching; ++j) {
			DistanceType new_sq_dist = m_distance(
					m_dataset.row(indices[i]).data, dcenters.row(j).data,
					m_veclen);
			if (sq_dist > new_sq_dist) {
				belongs_to[i] = j;
				sq_dist = new_sq_dist;
			}
		}
		count[belongs_to[i]]++;
	}

	bool converged = false;
	int iteration = 0;
	while (!converged && iteration < m_iterations) {
		converged = true;
		iteration++;

		//TODO: computeCentroids compute the new cluster centers
		// Warning: using matrix of integers, there might be an overflow when summing too much descriptors
		cv::Mat bitwiseCount(m_branching, m_veclen * 8,
				cv::DataType<int>::type);
		// Zeroing matrix of cumulative bits
		bitwiseCount = cv::Scalar::all(0);
		// Zeroing all the centroids dimensions
		dcenters = cv::Scalar::all(0);

		// Bitwise summing the data into each centroid
		for (size_t i = 0; (int) i < indices_length; i++) {
			uint j = belongs_to[i];
			cv::Mat b = bitwiseCount.row(j);
			KMajorityIndex::cumBitSum(m_dataset.row(indices[i]), b);
		}
		// Bitwise majority voting
		for (size_t j = 0; (int) j < m_branching; j++) {
			cv::Mat centroid = dcenters.row(j);
			KMajorityIndex::majorityVoting(bitwiseCount.row(j), centroid,
					count[j]);
		}

		// TODO quantize: reassign points to clusters
		for (int i = 0; i < indices_length; ++i) {
			DistanceType sq_dist = m_distance(m_dataset.row(indices[i]).data,
					dcenters.row(0).data, m_veclen);
			int new_centroid = 0;
			for (int j = 1; j < m_branching; ++j) {
				DistanceType new_sq_dist = m_distance(
						m_dataset.row(indices[i]).data, dcenters.row(j).data,
						m_veclen);
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

	TDescriptor** centers = new TDescriptor*[m_branching];

	for (int i = 0; i < m_branching; ++i) {
		centers[i] = new TDescriptor[m_veclen];
		m_memoryCounter += (int) (m_veclen * sizeof(TDescriptor));
		for (size_t k = 0; k < m_veclen; ++k) {
			centers[i][k] = dcenters.at<TDescriptor>(i, k);
		}
	}

	// compute kmeans clustering for each of the resulting clusters
	node->children = m_pool.allocate<VocabTreeNodePtr>(m_branching);
	int start = 0;
	int end = start;
	for (int c = 0; c < m_branching; ++c) {
		// Re-order indices by chunks in clustering order
		for (int i = 0; i < indices_length; ++i) {
			if (belongs_to[i] == c) {
				std::swap(indices[i], indices[end]);
				std::swap(belongs_to[i], belongs_to[end]);
				end++;
			}
		}

		node->children[c] = m_pool.allocate<VocabTreeNode>();
		node->children[c]->center = centers[c];
//		node->children[c]->indices = NULL;
		computeClustering(node->children[c], indices + start, end - start,
				level + 1);
		start = end;
	}

	dcenters.release();
	delete[] centers;
	delete[] count;
	delete[] belongs_to;
}

// --------------------------------------------------------------------------

void VocabTree::setNodeWeights(const std::vector<cv::Mat>& training_matrices,
		DBoW2::WeightingType weighting) {
	const uint NWords = m_words.size();
	const uint NDocs = training_matrices.size();

	if (weighting == DBoW2::TF || weighting == DBoW2::BINARY) {
		// IDF part must be 1 always
		for (size_t i = 0; i < NWords; i++) {
			m_words[i]->weight = 1;
		}
	} else if (weighting == DBoW2::IDF || weighting == DBoW2::TF_IDF) {
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
			for (size_t i = 0; (int) i < training_data.rows; i++) {
				uint word_id;
				double weight;
				quantize(training_data.row(i), word_id, weight);
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

void VocabTree::quantize(const cv::Mat& features, DBoW2::BowVector& v,
		DBoW2::WeightingType weighting, DBoW2::ScoringType scoring) const {

	if (features.type() != CV_8U) {
		throw std::runtime_error(
				"BHCIndex::quantize: error, features matrix is not binary\n");
	}

	if (features.cols != (int) m_veclen) {
		std::stringstream msg;
		msg << "BHCIndex::quantize: error, features vectors must be "
				<< m_veclen << " bytes long, that is " << m_veclen * 8
				<< "-dimensional\n";
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
	bool must = createScoringObject(scoring)->mustNormalize(norm);

	if (weighting == DBoW2::TF || weighting == DBoW2::TF_IDF) {
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

void VocabTree::quantize(const cv::Mat &features, uint &word_id,
		double &weight) const {

	VocabTreeNodePtr best_node = m_root;

	while (best_node->children != NULL) {

		VocabTreeNodePtr node = best_node;

		// Arbitrarily assign to first child
		best_node = node->children[0];
		DistanceType best_distance = m_distance(features.data,
				best_node->center, m_veclen);

		// Looking for a better child
		for (size_t j = 1; (int) j < this->m_branching; j++) {
			DistanceType d = m_distance(features.data,
					node->children[j]->center, m_veclen);
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

bool VocabTree::empty() const {
	return m_words.empty();
}

// --------------------------------------------------------------------------

double VocabTree::score(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2,
		DBoW2::ScoringType m_scoring) const {

	cv::Ptr<DBoW2::GeneralScoring> m_scoring_object = createScoringObject(
			m_scoring);

	return m_scoring_object->score(v1, v2);
}

// --------------------------------------------------------------------------

cv::Ptr<DBoW2::GeneralScoring> VocabTree::createScoringObject(
		DBoW2::ScoringType scoring) const {

	cv::Ptr<DBoW2::GeneralScoring> m_scoring_object = NULL;

	switch (scoring) {
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

	return m_scoring_object;
}

} /* namespace cvflann */
