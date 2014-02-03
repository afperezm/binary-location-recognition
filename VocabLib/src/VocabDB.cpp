/*
 * VocabDB.cpp
 *
 *  Created on: Jan 26, 2014
 *      Author: andresf
 */

#include <VocabDB.hpp>

namespace vlr {

void VocabDB::saveInvertedIndex(const std::string& filename) const {
	m_invertedIndex->save(filename);
}

// --------------------------------------------------------------------------

void VocabDB::loadInvertedIndex(const std::string& filename) {
	m_invertedIndex->load(filename);
}

// --------------------------------------------------------------------------

void VocabDB::addImageToDatabase(int dbImgIdx, cv::Mat dbImgFeatures) {

	int m_veclen = getFeaturesLength();

	if (dbImgFeatures.empty() == false && dbImgFeatures.cols != m_veclen) {
		std::stringstream ss;
		ss << "Error while adding image, feature vector has different length"
				" than the ones used for building the tree, it is ["
				<< dbImgFeatures.cols << "] while it should be[" << m_veclen
				<< "]";
		throw std::runtime_error(ss.str());
	}

	if (m_invertedIndex->empty() == true) {
		throw std::runtime_error(
				"[VocabDB::addImageToDatabase] Error while adding image,"
						" vocabulary is empty");
	}

	int wordId;
	double wordWeight; // not needed

	for (int i = 0; i < dbImgFeatures.rows; ++i) {
		quantize(dbImgFeatures.row(i), wordId, wordWeight);
		m_invertedIndex->addFeatureToInvertedFile(wordId, dbImgIdx);
	}

	// Increasing the counter of images in the DB
	++m_invertedIndex->m_numDbImages;
}

// --------------------------------------------------------------------------

void VocabDB::computeWordsWeights(vlr::WeightingType weighting) {

	if (m_invertedIndex->empty()) {
		throw std::runtime_error("[VocabDB::computeWordsWeights]"
				" Error while computing words weights, vocabulary is empty");
	}

	if (weighting == vlr::TF) {
		// Setting constant weight equal to 1
		for (vlr::Word& word : *m_invertedIndex) {
			word.m_weight = 1.0;
		}
	} else if (weighting == vlr::TF_IDF) {
		// Calculating the IDF part of the TF-IDF score, the complete
		// TF-IDF score is the result of multiplying the weight by the word count
		for (vlr::Word& word : *m_invertedIndex) {
			int len = word.m_imageList.size();
			// because having that a descriptor from all DB images is quantized
			// to the same word is quite unlikely
			if (len > 0) {
				word.m_weight = log(
						(double) m_invertedIndex->m_numDbImages / (double) len);
			} else {
				word.m_weight = 0.0;
			}
		}
	} else {
		throw std::runtime_error(
				"[VocabDB::computeWordsWeights] Unknown weighting type");
	}
}

// --------------------------------------------------------------------------

void VocabDB::createDatabase() {

	if (m_invertedIndex->empty()) {
		throw std::runtime_error("[VocabDB::createDatabase] Error while"
				" applying weights to words histogram, vocabulary is empty");
	}

	// Loop over words
	for (vlr::Word& word : *m_invertedIndex) {
		// Apply word weight to the image count
		for (vlr::ImageCount& image : word.m_imageList) {
			image.m_count *= word.m_weight;
		}
	}

}

// --------------------------------------------------------------------------

void VocabDB::normalizeDatabase(int normType) {

	if (m_invertedIndex->empty() == true) {
		throw std::runtime_error("[VocabDB::normalizeDatabase] Error while"
				" normalizing DB BoF vectors, vocabulary is empty");
	}

	// Magnitude of a vector is defined as: sum(abs(xi)^p)^(1/p)

	std::vector<float> mags(m_invertedIndex->m_numDbImages, 0.0);

	// Computing DB BoF vectors magnitude

	// Summing vector elements
	for (vlr::Word& word : *m_invertedIndex) {
		for (vlr::ImageCount& image : word.m_imageList) {
			uint index = image.m_index;
			double dim = image.m_count;

			CV_Assert(index < mags.size());

			if (normType == cv::NORM_L1) {
				mags[index] += fabs(dim);
			} else if (normType == cv::NORM_L2) {
				mags[index] += pow(dim, 2);
			} else {
				throw std::runtime_error(
						"[VocabDB::normalizeDatabase] Unknown scoring method");
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
	for (vlr::Word& word : *m_invertedIndex) {
		for (vlr::ImageCount& image : word.m_imageList) {
			uint index = image.m_index;
			assert(index < mags.size());
			if (mags[index] > 0.0) {
				image.m_count /= mags[index];
			}
		}
	}

}

// --------------------------------------------------------------------------

void VocabDB::clearDatabase() {
	m_invertedIndex->resize(getNumOfWords(), vlr::Word(1.0));
	for (vlr::Word& word : *m_invertedIndex) {
		std::vector<vlr::ImageCount>().swap(word.m_imageList);
	}
}

// --------------------------------------------------------------------------

void VocabDB::scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
		const int normType) const {

	int m_veclen = getFeaturesLength();

	if (queryImgFeatures.rows < 1) {
		throw std::runtime_error(
				"[VocabDB::scoreQuery] Error while scoring image, at least one feature vector is needed");
	}

	if (queryImgFeatures.cols != m_veclen) {
		std::stringstream ss;
		ss << "Error while adding image, feature vector has different length"
				" than the ones used for building the tree, it is ["
				<< queryImgFeatures.cols << "] while it should be[" << m_veclen
				<< "]";
		throw std::runtime_error(ss.str());
	}

	if (m_invertedIndex->empty() == true) {
		throw std::runtime_error("[VocabDB::scoreQuery]"
				" Error while scoring query, vocabulary is empty");
	}

	if (normType != cv::NORM_L1 && normType != cv::NORM_L2) {
		throw std::runtime_error(
				"[VocabDB::scoreQuery] Unknown scoring method");
	}

	scores = cv::Mat::zeros(1, m_invertedIndex->m_numDbImages,
			cv::DataType<float>::type);

	cv::Mat queryBoFVector;
	transform(queryImgFeatures, queryBoFVector, normType);

	//	Efficient scoring query BoF vector against all DB BoF vectors

	// ||v - w||_{L1} = 2 + Sum(|v_i - w_i| - |v_i| - |w_i|)
	// ||v - w||_{L2} = sqrt( 2 - 2 * Sum(v_i * w_i) )

	// Calculating sum part of the efficient score implementation
	for (size_t wordId = 0; wordId < m_invertedIndex->size(); ++wordId) {
		float qi = queryBoFVector.at<float>(0, wordId);

		// Early exit
		if (qi == 0.0) {
			continue;
		}

		// The inverted file of a word contains all images counts quantized into that word
		// i.e. if they are there its because their count di is not zero

		// In addition its fair computing qi against di without further verification
		// since the inverted files contain not null counts

		for (int imageId = 0;
				imageId < int(m_invertedIndex->at(wordId).m_imageList.size());
				++imageId) {
			float di = m_invertedIndex->at(wordId).m_imageList[imageId].m_count;

			// qi cannot be zero because we are considering only when its non-zero
			// qi cannot be more than 1 because it is supposed to be normalized
			CV_Assert(qi > 0 && qi <= 1.0);

			// di cannot more than 1 because it is supposed to be normalized
			CV_Assert(di <= 1.0);

			// di cannot be zero (unless the weight is zero) because the inverted files
			// contain only counts for images with a descriptor which was quantized
			// into that word
			if (m_invertedIndex->at(wordId).m_weight != 0.0) {
				CV_Assert(di > 0.0);
			} else {
				CV_Assert(di >= 0.0);
			}

			if (normType == cv::NORM_L1) {
				scores.at<float>(0,
						m_invertedIndex->at(wordId).m_imageList[imageId].m_index) +=
						(float) (fabs(qi - di) - fabs(qi) - fabs(di));
			} else if (normType == cv::NORM_L2) {
				scores.at<float>(0,
						m_invertedIndex->at(wordId).m_imageList[imageId].m_index) +=
						(float) qi * di;
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

// --------------------------------------------------------------------------

void VocabDB::transform(const cv::Mat& featuresVector, cv::Mat& bofVector,
		const int normType) const {

	// Initialize query BoF vector
	bofVector = cv::Mat::zeros(1, m_invertedIndex->size(),
			cv::DataType<float>::type);

	int wordIdx;
	double wordWeight;

	int numInvertedFiles = m_invertedIndex->size();

	// Quantize each query image feature vector
	for (int i = 0; i < featuresVector.rows; ++i) {

		quantize(featuresVector.row(i), wordIdx, wordWeight);

		if (wordIdx > numInvertedFiles - 1) {
			throw std::runtime_error(
					"[VocabDB::transform] Feature quantized into a non-existent word");
		}

		bofVector.at<float>(0, wordIdx) += (float) wordWeight;
	}

	//	Normalizing query BoW vector
	cv::normalize(bofVector, bofVector, 1, 0, normType);

}

// --------------------------------------------------------------------------

void VocabDB::getDatabaseBoFVector(unsigned int dbImgIdx,
		cv::Mat& dbBoFVector) const {

	if (m_invertedIndex->empty() == true) {
		throw std::runtime_error(
				"[VocabDB::getDbBoFVector] Error while obtaining DB BoF vectors,"
						" vocabulary is empty");
	}

	dbBoFVector = cv::Mat::zeros(1, m_invertedIndex->size(),
			cv::DataType<float>::type);

	for (int wordId = 0; wordId < int(m_invertedIndex->size()); ++wordId) {
		for (int imageId = 0;
				imageId < int(m_invertedIndex->at(wordId).m_imageList.size());
				++imageId) {
			if (m_invertedIndex->at(wordId).m_imageList[imageId].m_index
					== dbImgIdx) {
				dbBoFVector.at<float>(0, wordId) =
						m_invertedIndex->at(wordId).m_imageList[imageId].m_count;
			}
		}
	}
}

// --------------------------------------------------------------------------

int HKMDB::getFeaturesLength() const {
	return m_bofModel->getVeclen();
}

// --------------------------------------------------------------------------

void HKMDB::quantize(const cv::Mat& feature, int& wordId,
		double& wordWeight) const {

	wordId = -1;
	int nodeAtL = -1;

	m_bofModel->quantize(feature, m_directIndex->getLevel(), wordId, nodeAtL);

	CV_Assert(wordId != -1);
	// Commented since not needed ==> using linear search in GeomVerify
	// CV_Assert(nodeAtL != -1);

	wordWeight = m_invertedIndex->at(wordId).m_weight;

}

// --------------------------------------------------------------------------

void HKMDB::loadBoFModel(const std::string& filename) {
	m_bofModel->load(filename);
	setDirectIndexLevel(m_levelsUp);
}

// --------------------------------------------------------------------------

size_t HKMDB::getNumOfWords() const {
	return m_bofModel->getWordsCount();
}

// --------------------------------------------------------------------------

int HKMDB::getDirectIndexLevel() {
	return m_directIndex->getLevel();
}

// --------------------------------------------------------------------------

void HKMDB::setDirectIndexLevel(int levelsUp) {
	/*
	 * TODO Implement method in VocabTree to retrieve effective depth since it
	 * might be different than the theoretical one due to the clustering process.
	 * This would result in a wrongly computed direct index level.
	 */
	int directIndexLevel = 0;

	if (levelsUp > m_bofModel->getDepth() - 1) {
		directIndexLevel = 0;
	} else if (levelsUp < 0) {
		directIndexLevel = m_bofModel->getDepth() - 1;
	} else {
		directIndexLevel = m_bofModel->getDepth() - 1 - levelsUp;
	}

	m_directIndex->setLevel(directIndexLevel);
}

// --------------------------------------------------------------------------

void HKMDB::saveDirectIndex(const std::string& filename) const {
	m_directIndex->save(filename);
}

// --------------------------------------------------------------------------

void HKMDB::loadDirectIndex(const std::string& filename) {
	m_directIndex->clear();
	m_directIndex->load(filename);
}

// --------------------------------------------------------------------------

int AKMajDB::getFeaturesLength() const {
	return bofModel->getCentroids().cols;
}

// --------------------------------------------------------------------------

void AKMajDB::loadBoFModel(const std::string& filename) {
	bofModel->load(filename);
}

// --------------------------------------------------------------------------

size_t AKMajDB::getNumOfWords() const {
	return bofModel->getCentroids().rows;
}

// --------------------------------------------------------------------------

void AKMajDB::quantize(const cv::Mat& feature, int& wordId,
		double& wordWeight) const {

	cvflann::LinearIndexParams params;

	// Find first exact nearest neighbor
	cv::Ptr<cvflann::NNIndex<cvflann::Hamming<uchar> > > linearIndex =
			cvflann::create_index_by_type(
					cvflann::Matrix<uchar>(
							(uchar*) bofModel->getCentroids().data,
							bofModel->getCentroids().rows,
							bofModel->getCentroids().cols), params,
					cvflann::Hamming<uchar>());

	int knn = 1;

	cvflann::Matrix<int> indices(new int[1 * knn], 1, knn);
	std::fill(indices.data, indices.data + indices.rows * indices.cols, 0);

	cvflann::Matrix<int> distances(new int[1 * knn], 1, knn);
	std::fill(distances.data, distances.data + distances.rows * distances.cols,
			0.0f);

	linearIndex->knnSearch(
			cvflann::Matrix<uchar>((uchar*) feature.data, 1, feature.cols),
			indices, distances, knn, cvflann::SearchParams());

	// Save word id and weight
	wordId = indices[0][0];
	wordWeight = m_invertedIndex->at(wordId).m_weight;

	delete[] indices.data;
	delete[] distances.data;

}

} /* namespace vlr */
