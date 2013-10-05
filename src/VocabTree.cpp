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
VocabTree::VocabTree(const cv::Mat& inputData = cv::Mat(),
		const IndexParams& params) :
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
		throw std::runtime_error(
				"[VocabTree::build] Error, branching factor must be at least 2");
	}

	// Number of features in the dataset
	size_t size = m_dataset.rows;

	//  Array of indices to vectors in the dataset
	int* indices = new int[size];
	for (size_t i = 0; i < size; ++i) {
		indices[i] = int(i);
	}

	printf("[VocabTree::build] Building tree from %d features\n", (int) size);
	printf("[VocabTree::build]   with depth %d, branching factor %d\n", m_depth,
			m_branching);
	printf("[VocabTree::build]   and restarts %d\n", m_iterations);

	m_root = m_pool.allocate<VocabTreeNode>();
	computeNodeStatistics(m_root, indices, (int) size);
	computeClustering(m_root, indices, (int) size, 0);
}

// --------------------------------------------------------------------------

void VocabTree::save(const std::string& filename) const {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::WRITE);

	if (!fs.isOpened()) {
		throw std::runtime_error(
				"[VocabTree::save] Error opening file " + filename
						+ " for writing");
	}

//	fs << "centersInit" << m_centers_init;
	fs << "iterations" << m_iterations;
	fs << "branching" << m_branching;
	fs << "depth" << m_depth;
	fs << "vectorLength" << (int) m_veclen;
	fs << "memoryCounter" << m_memoryCounter;

	fs << "root";

	save_tree(fs, m_root);

	fs.release();
}

// --------------------------------------------------------------------------

void VocabTree::save_tree(cv::FileStorage& fs, VocabTreeNodePtr node) const {

	// WriteNode
	fs << "{";
	fs << "center" << cv::Mat(1, m_veclen, m_dataset.type(), node->center);
	fs << "weight" << node->weight;
	fs << "wordId" << node->word_id;
	fs << "imageList" << "[";
	for (ImageCount img : node->image_list) {
		fs << "{:";
		fs << "m_index" << (int) img.m_index;
		fs << "m_count" << img.m_count;
		fs << "}";
	}
	fs << "]";
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

void VocabTree::load(const std::string& filename) {

	cv::FileStorage fs(filename.c_str(), cv::FileStorage::READ);

	if (!fs.isOpened()) {
		throw std::runtime_error(
				std::string("Could not open file ") + filename);
	}

	m_iterations = (int) fs["iterations"];
	m_branching = (int) fs["branching"];
	m_depth = (int) fs["depth"];
	m_veclen = (int) fs["vectorLength"];
	m_memoryCounter = (int) fs["memoryCounter"];

	cv::FileNode root = fs["root"];
	m_root = m_pool.allocate<VocabTreeNode>();
	load_tree(root, m_root);

	fs.release();
}

// --------------------------------------------------------------------------

void VocabTree::load_tree(cv::FileNode& fs, VocabTreeNodePtr& node) {

	cv::Mat center;
	fs["center"] >> center;
	node->center = center.data;
	node->weight = (double) fs["weight"];
	node->word_id = (int) fs["wordId"];

	cv::FileNode children = fs["children"];

	// Verifying that the retrieved children collection has 0 or k elements
	CV_Assert(
			children.type() == cv::FileNode::SEQ
					&& (children.size() == 0
							|| (int ) children.size() == m_branching));

	if (children.size() == 0) {
		// Node has no children then it's a leaf node
		cv::FileNode images = fs["imageList"];
		node->image_list.clear();
		for (cv::FileNodeIterator it = images.begin(); it != images.end();
				++it) {
			size_t index = (int) (*it)["m_index"];
			ImageCount* img = new ImageCount(index, (float) (*it)["m_count"]);
			node->image_list.push_back(*img);
		}
	} else {
		// Node has children then it's an interior node
		node->children = m_pool.allocate<VocabTreeNodePtr>(m_branching);
		cv::FileNodeIterator it = children.begin();

		for (size_t c = 0; (int) c < m_branching; ++c) {
			node->children[c] = m_pool.allocate<VocabTreeNode>();
			cv::FileNode child = *it;
			load_tree(child, node->children[c]);
			it++;
		}
	}

}

// --------------------------------------------------------------------------

int VocabTree::usedMemory() const {
	return m_pool.usedMemory + m_pool.wastedMemory + m_memoryCounter;
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

// Recursion base case: done when the last level is reached
// or when there are less data than clusters
	if (level == m_depth - 1 || indices_length < m_branching) {
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

// compute k-means clustering for each of the resulting clusters
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

//void VocabTree::setNodeWeights(const std::vector<cv::Mat>& training_matrices,
//		DBoW2::WeightingType weighting) {
//	const uint NWords = m_words.size();
//	const uint NDocs = training_matrices.size();
//
//	if (weighting == DBoW2::TF || weighting == DBoW2::BINARY) {
//		// IDF part must be 1 always
//		for (size_t i = 0; i < NWords; i++) {
//			m_words[i]->weight = 1;
//		}
//	} else if (weighting == DBoW2::IDF || weighting == DBoW2::TF_IDF) {
//		// IDF and TF-IDF: we calculate the IDF path now
//
//		// Note: this actually calculates the IDF part of the TF-IDF score.
//		// The complete TF-IDF score is calculated in ::transform
//
//		// TODO When test see if this is the same as the inverted file length,
//		// for me it looks like it does
//
//		// Ni: number of documents/images in which the ith words appears
//		std::vector<uint> Ni(NWords, 0);
//		std::vector<bool> counted(NWords, false);
//
//		for (cv::Mat training_data : training_matrices) {
//			// Restart word count 'cause new image features matrix
//			std::fill(counted.begin(), counted.end(), false);
//			for (size_t i = 0; (int) i < training_data.rows; i++) {
//				uint word_id;
//				// Obtaining id of the word where ith feature vector is quantized
//				double weight;
//				quantize(training_data.row(i), word_id, weight);
//				// Count only once the appearance of the word in the image (training matrix)
//				if (!counted[word_id]) {
//					Ni[word_id]++;
//					counted[word_id] = true;
//				}
//			}
//		}
//
//		// Set ln(N/Ni)
//		for (size_t i = 0; i < NWords; i++) {
//			if (Ni[i] > 0) {
//				m_words[i]->weight = log((double) NDocs / (double) Ni[i]);
//			}
//			// TODO else: this cannot occur if using kmeans++
//		}
//	}
//}

// --------------------------------------------------------------------------

//void VocabTree::transform(const cv::Mat& features, DBoW2::BowVector& v,
//		DBoW2::WeightingType weighting, DBoW2::ScoringType scoring) const {
//
//	if (features.type() != CV_8U) {
//		throw std::runtime_error(
//				"[VocabTree::quantize] Features matrix is not binary");
//	}
//
//	if (features.cols != (int) m_veclen) {
//		std::stringstream ss;
//		ss << "[VocabTree::quantize] Features vectors must be " << m_veclen
//				<< " bytes long, i.e. " << m_veclen * 8 << "-dimensional";
//		throw std::runtime_error(ss.str());
//	}
//
//	if (features.rows < 1) {
//		throw std::runtime_error(
//				"[VocabTree::quantize] At least one feature vector is needed");
//	}
//
//	v.clear();
//
//	if (empty()) {
//		return;
//	}
//
//	for (size_t i = 0; (int) i < features.rows; i++) {
//		uint id;
//		double w;
//		// w is the IDF value if TF_IDF, 1 if TF
//		// w is the inverse document frequency if IDF, or 1 if BINARY
//		quantize(features.row(i), id, w);
//		// not stopped
//		if (w > 0) {
//			if (weighting == DBoW2::TF || weighting == DBoW2::TF_IDF) {
//				// TF or TF-IDF
//				v.addWeight(id, w);
//			} else {
//				// IDF or BINARY
//				v.addIfNotExist(id, w);
//			}
//		}
//	}
//
//	// Flag indicating normalization
//	DBoW2::LNorm norm;
//	bool must = createScoringObject(scoring)->mustNormalize(norm);
//
//	// Normalizing vector by its length when using TF or TF-IDF for weighting
//	if ((weighting == DBoW2::TF || weighting == DBoW2::TF_IDF)
//			&& (!v.empty() && !must)) {
//		const double nd = v.size();
//
//		for (DBoW2::BowVector::iterator vit = v.begin(); vit != v.end();
//				vit++) {
//			vit->second /= nd;
//		}
//	}
//
//	if (must) {
//		v.normalize(norm);
//	}
//}

// --------------------------------------------------------------------------

void VocabTree::quantize(const cv::Mat& feature, uint &word_id,
		double &weight) const {

	VocabTreeNodePtr best_node = m_root;

	while (best_node->children != NULL) {

		VocabTreeNodePtr node = best_node;

		// Arbitrarily assign to first child
		best_node = node->children[0];
		DistanceType best_distance = m_distance(feature.data, best_node->center,
				m_veclen);

		// Looking for a better child
		for (size_t j = 1; (int) j < this->m_branching; j++) {
			DistanceType d = m_distance(feature.data, node->children[j]->center,
					m_veclen);
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

void VocabTree::computeWordsWeights(const uint numDbWords,
		DBoW2::WeightingType weighting) {
	if (weighting == DBoW2::TF || weighting == DBoW2::BINARY) {
		// Setting constant weight equal to 1
		for (VocabTreeNodePtr word : m_words) {
			word->weight = 1;
		}
	} else if (weighting == DBoW2::IDF || weighting == DBoW2::TF_IDF) {
		// Calculating the IDF part of the TF-IDF score, the complete
		// TF-IDF score is the result of multiplying the weight by the word count
		for (VocabTreeNodePtr word : m_words) {
			int len = (int) word->image_list.size();
			if (len > 0) {
				word->weight = log((double) numDbWords / (double) len);
			} else {
				word->weight = 0.0;
			}
		}
	}
}

// --------------------------------------------------------------------------

void VocabTree::createDatabase() {

	// Loop over words
	for (VocabTreeNodePtr word : m_words) {
		// Apply word weight to the image count
		for (ImageCount image : word->image_list) {
			image.m_count *= word->weight;
		}
	}

}

// --------------------------------------------------------------------------

void VocabTree::clearDatabase() {
	for (VocabTreeNodePtr word : m_words) {
		word->image_list.clear();
	}
}

// --------------------------------------------------------------------------

bool VocabTree::empty() const {
	return m_words.empty();
}

// --------------------------------------------------------------------------

//double VocabTree::score(const DBoW2::BowVector &v1, const DBoW2::BowVector &v2,
//		DBoW2::ScoringType m_scoring) const {
//
//	cv::Ptr<DBoW2::GeneralScoring> m_scoring_object = createScoringObject(
//			m_scoring);
//
//	return m_scoring_object->score(v1, v2);
//}

// --------------------------------------------------------------------------

//cv::Ptr<DBoW2::GeneralScoring> VocabTree::createScoringObject(
//		DBoW2::ScoringType scoring) const {
//
//	cv::Ptr<DBoW2::GeneralScoring> m_scoring_object = NULL;
//
//	switch (scoring) {
//	case DBoW2::L1_NORM:
//		m_scoring_object = new DBoW2::L1Scoring;
//		break;
//
//	case DBoW2::L2_NORM:
//		m_scoring_object = new DBoW2::L2Scoring;
//		break;
//
//	case DBoW2::CHI_SQUARE:
//		m_scoring_object = new DBoW2::ChiSquareScoring;
//		break;
//
//	case DBoW2::KL:
//		m_scoring_object = new DBoW2::KLScoring;
//		break;
//
//	case DBoW2::BHATTACHARYYA:
//		m_scoring_object = new DBoW2::BhattacharyyaScoring;
//		break;
//
//	case DBoW2::DOT_PRODUCT:
//		m_scoring_object = new DBoW2::DotProductScoring;
//		break;
//
//	}
//
//	return m_scoring_object;
//}

// --------------------------------------------------------------------------

void VocabTree::addImageToDatabase(uint imgIdx, cv::Mat imgFeatures) {

	if (imgFeatures.type() != CV_8U) {
		throw std::runtime_error(
				"[VocabTree::addImageToDatabase] Features matrix is not binary");
	}

	if (imgFeatures.cols != (int) m_veclen) {
		std::stringstream ss;
		ss << "[VocabTree::addImageToDatabase] Features vectors must be "
				<< m_veclen << " bytes long, i.e. " << m_veclen * 8
				<< "-dimensional";
		throw std::runtime_error(ss.str());
	}

	if (imgFeatures.rows < 1) {
		throw std::runtime_error(
				"[VocabTree::addImageToDatabase] At least one feature vector is needed");
	}

	if (empty()) {
		throw std::runtime_error(
				"[VocabTree::addImageToDatabase] Vocabulary is empty");
	}

	for (size_t i = 0; (int) i < imgFeatures.rows; i++) {
		uint wordIdx;
		double wordWeight;
		// w is the IDF value if TF_IDF, 1 if TF
		// w is the Inverse Document Frequency if IDF, 1 if BINARY
		quantize(imgFeatures.row(i), wordIdx, wordWeight);
		addFeatureToInvertedFile(wordIdx, imgIdx);
	}

}

// --------------------------------------------------------------------------

void VocabTree::addFeatureToInvertedFile(uint wordIdx, uint imgIdx) {

	int n = (int) m_words[wordIdx]->image_list.size();

	// First time a new image must be pushed
	if (n == 0) {
		m_words[wordIdx]->image_list.push_back(
				ImageCount(imgIdx, (float) m_words[wordIdx]->weight));
	} else {
		// Images list is not empty: check if the id of the last added image
		// is the same than that of the image being added
		if (m_words[wordIdx]->image_list[n - 1].m_index == imgIdx) {
			// Images are equal then the counter is increased by one
			m_words[wordIdx]->image_list[n - 1].m_count += 1;
		} else {
			// Images are different then push a new image
			m_words[wordIdx]->image_list.push_back(
					ImageCount(imgIdx, (float) m_words[wordIdx]->weight));
		}
	}

}

// --------------------------------------------------------------------------

void VocabTree::normalizeDatabase(const uint num_db_images, int normType) {

	// Magnitude of a vector is defined as: sum(abs(xi)^p)^(1/p)

	std::vector<float> mags;
	mags.resize(num_db_images);

	// Loop over words
	for (VocabTreeNodePtr word : m_words) {
		for (ImageCount image : word->image_list) {
			uint index = image.m_index;
			double dim = image.m_count;

			assert(index < mags.size());

			if (normType == cv::NORM_L1) {
				mags[index] += fabs(dim);
			} else if (normType == cv::NORM_L2) {
				mags[index] += pow(dim, 2);
			} else {
				throw std::runtime_error(
						"[VocabTree::scoreQuery] Unknown scoring method");
			}
		}
	}

	// Print magnitudes
	for (size_t i = 0; i < num_db_images; i++) {
		if (normType == cv::NORM_L2) {
			mags[i] = sqrt(mags[i]);
		}
//		printf("[VocabTree::normalizeDatabase] Vector %d has magnitude %0.3f\n", i, mags[i]);
	}

	// Normalizing database
	for (VocabTreeNodePtr word : m_words) {
		for (ImageCount image : word->image_list) {
			uint index = image.m_index;
			assert(index < mags.size());
			if (mags[index] > 0.0) {
				image.m_count /= mags[index];
			}
		}
	}

}

// --------------------------------------------------------------------------

void VocabTree::scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
		const uint numDbImages, const int normType) const {

	if (queryImgFeatures.type() != CV_8U) {
		throw std::runtime_error(
				"[VocabTree::scoreQueryFeatures] Features matrix is not binary");
	}

	if (queryImgFeatures.cols != (int) m_veclen) {
		std::stringstream ss;
		ss << "[VocabTree::scoreQueryFeatures] Features vectors must be "
				<< m_veclen << " bytes long, i.e. " << m_veclen * 8
				<< "-dimensional";
		throw std::runtime_error(ss.str());
	}

	if (queryImgFeatures.rows < 1) {
		throw std::runtime_error(
				"[VocabTree::scoreQueryFeatures] At least one feature vector is needed");
	}

	if (empty()) {
		throw std::runtime_error(
				"[VocabTree::scoreQueryFeatures] Vocabulary is empty");
	}

	scores = cv::Mat();
	scores = cv::Mat(1, numDbImages, cv::DataType<float>::type, 0);
	cv::Mat queryBowVector(1, m_words.size(), cv::DataType<float>::type, 0);

	// Quantize each query image feature vector
	for (size_t i = 0; (int) i < queryImgFeatures.rows; i++) {
		uint wordIdx;
		double wordWeight;
		quantize(queryImgFeatures.row(i), wordIdx, wordWeight);

		queryBowVector.at<float>(1, wordIdx) += wordWeight;
	}

	cv::normalize(queryBowVector, queryBowVector, 0, 1, cv::NORM_L1);

	// ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
	// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )

	// Calculating sum part of the efficient score implementation
	for (VocabTreeNodePtr word : m_words) {
		// Early exit
		if (queryBowVector.at<float>(1, word->word_id) == 0.0) {
			continue;
		}

		for (ImageCount image : word->image_list) {
			double vi = queryBowVector.at<float>(1, word->word_id);
			double wi = image.m_count;
			if (normType == cv::NORM_L1) {
				scores.at<float>(1, image.m_index) += fabs(vi - wi) - fabs(vi)
						- fabs(wi);
			} else if (normType == cv::NORM_L2) {
				scores.at<float>(1, image.m_index) += vi * wi;
			} else {
				throw std::runtime_error(
						"[VocabTree::scoreQuery] Unknown scoring method");
			}
		}
	}

	// Completing efficient score implementation
	for (int i = 0; i < scores.cols; i++) {
		if (normType == cv::NORM_L1) {
			scores.at<float>(1, i) = 2 + scores.at<float>(1, i);
		} else if (normType == cv::NORM_L2) {
			scores.at<float>(1, i) = 2 - 2 * scores.at<float>(1, i);
		} else {
			throw std::runtime_error(
					"[VocabTree::scoreQuery] Unknown scoring method");
		}
	}

}

} /* namespace cvflann */
