/*
 * VocabDB.h
 *
 *  Created on: Jan 26, 2014
 *      Author: andresf
 */

#ifndef VOCABDB_H_
#define VOCABDB_H_

#include <KMajority.h>
#include <InvertedIndex.hpp>
#include <VocabTree.h>

namespace vlr {

class VocabDB {

protected:

	vlr::InvertedIndex m_invertedIndex;

public:

	/**
	 * Virtual destroyer to enable destruction from a subclass
	 */
	virtual ~VocabDB(){
	}

	virtual int getFeaturesLength() const = 0;

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
	 * Quantizes a single feature vector into a word and stores
	 * the resulting word id and weight.
	 *
	 * @param feature - Row vector representing the feature vector to quantize
	 * @param wordId - The id of the found word
	 * @param wordWeight - The weight of the found word
	 */
	virtual void quantize(const cv::Mat& feature, int& wordId,
			double& wordWeight) const = 0;

	/**
	 * Quantizes DB image features into the vocabulary and updates the inverted file.
	 *
	 * @param dbImgIdx - The id of the image
	 * @param dbImgFeatures - Matrix of features representing the image
	 */
	void addImageToDatabase(int dbImgIdx, cv::Mat dbImgFeatures);

	/**
	 * Assigns weights to the vocabulary words by applying the chosen
	 * weighting scheme to the entries on the inverted files.
	 *
	 * @param weighting - The weighting scheme to apply
	 */
	void computeWordsWeights(bfeat::WeightingType weighting);

	/**
	 * Computes the DB BoF vectors by applying the words weights
	 * to the image counts in the inverted files.
	 */
	void createDatabase();

	/**
	 * Normalizes the DB BoF vectors by dividing the weighted counts
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
	 * Computes the query BoF vector of an image by quantizing the query image
	 * features and applying the words weights, followed by efficiently scoring it against
	 * the pre-computed DB BoF vectors.
	 *
	 * @param queryImgFeatures - Matrix containing the features of the query image
	 * @param scores - Row matrix of size [1 x n] where n is the number DB images
	 * @param normType - normalization method used for scoring BoF vectors
	 *
	 * @note DB BoF vectors must be normalized beforehand
	 */
	void scoreQuery(const cv::Mat& queryImgFeatures, cv::Mat& scores,
			const int normType = cv::NORM_L2) const;

	/**
	 * Transforms a set of data (representing a single image) into a BoF vector.
	 *
	 * @param featuresVector - Matrix of data to quantize
	 * @param bofVector - BoF vector of weighted words
	 * @param normType - Norm used to normalize the output query BoF vector
	 */
	void transform(const cv::Mat& featuresVector, cv::Mat& bofVector,
			const int normType) const;

	/**
	 * Retrieves a DB BoF vector given its index.
	 *
	 * @param dbImgIdx - The index of the DB image
	 * @param dbBoFVector - A reference to the matrix where BoF vector will be save
	 */
	void getDbBoFVector(uint dbImgIdx, cv::Mat& dbBoFVector) const;

};

// --------------------------------------------------------------------------

class HKMDB: public VocabDB {

protected:

	// TODO Initialize bofModel at the constructor
	cv::Ptr<bfeat::VocabTreeBase> bofModel;

public:

	HKMDB() :
			bofModel(NULL) {
	}

	~HKMDB() {
	}

	int getFeaturesLength() const;

	void quantize(const cv::Mat& feature, int& wordId,
			double& wordWeight) const;

};

// --------------------------------------------------------------------------

class AKMajDB: public VocabDB {

protected:

	// TODO Initialize bofModel at the constructor
	cv::Ptr<KMajority> bofModel;

public:

	AKMajDB() :
			bofModel(NULL) {
	}

	~AKMajDB() {
	}

	int getFeaturesLength() const;

	void quantize(const cv::Mat& feature, int& wordId,
			double& wordWeight) const;
};

} /* namespace vlr */

#endif /* VOCABDB_H_ */
