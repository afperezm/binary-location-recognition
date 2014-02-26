/*
 * VocabBuildDB.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <sys/stat.h>

#include <boost/regex.hpp>

#include <opencv2/core/core.hpp>

#include <VocabTree.h>
#include <VocabDB.hpp>

#include <FileUtils.hpp>

double mytime;

int main(int argc, char **argv) {

	if (argc < 4 || argc > 7) {
		printf("\nUsage:\n\tVocabBuildDB <in.db.images.list> "
				"<in.vocab> <out.inverted.index>"
				" [in.weighting:TFIDF] [in.norm:L2] [out.nn.index]\n\n"
				"Weighting:\n"
				"\tTFIDF: Term Frequency - Inverse Document Frequency\n"
				"\tTF: Term Frequency\n"
				"\tBIN: Binary\n\n"
				"Norm:\n"
				"\tL1: L1-norm\n"
				"\tL2: L2-norm\t\n\n");
		return EXIT_FAILURE;
	}

	std::string in_list = argv[1];
	std::string in_vocab = argv[2];
	std::string out_inv_index = argv[3];
	std::string in_weighting = "TFIDF";
	std::string in_norm = "L1";
	std::string out_nn_index = "nn_index.bin";

	if (argc >= 5) {
		in_weighting = argv[4];
	}

	if (argc >= 6) {
		in_norm = argv[5];
	}

	if (argc >= 7) {
		out_nn_index = argv[6];
	}

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(in_vocab, expression) == false) {
		fprintf(stderr,
				"Input vocabulary file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	if (boost::regex_match(out_inv_index, expression) == false) {
		fprintf(stderr,
				"Output inverted index file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: read list of descriptors that shall be used to build the vocabulary
	printf("-- Loading list of database images descriptors\n");
	std::vector<std::string> descFilenames;
	FileUtils::loadList(in_list, descFilenames);
	printf("   Loaded, got [%lu] entries\n", descFilenames.size());

	printf("-- Building database using [%lu] images\n", descFilenames.size());

	cv::Ptr<vlr::VocabDB> db;

	std::string in_vocab_type = vlr::VocabBase::loadVocabType(in_vocab);

	if (in_vocab_type.compare("HKM") == 0) {
		// HKM
		db = new vlr::HKMDB(false);
	} else if (in_vocab_type.compare("HKMAJ") == 0) {
		// HKMaj
		db = new vlr::HKMDB(true);
	} else if (in_vocab_type.compare("AKMAJ") == 0) {
		// AKMaj
		db = new vlr::AKMajDB();
	} else {
		fprintf(stderr, "Vocabulary type [%s] is not valid\n", in_vocab_type);
		return EXIT_FAILURE;
	}

	printf("-- Reading vocabulary from [%s]\n", in_vocab.c_str());

	mytime = cv::getTickCount();
	db->loadBoFModel(in_vocab);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Vocabulary loaded in [%lf] ms, got [%lu] words \n", mytime,
			db->getNumOfWords());

	// Create and save nearest neighbor index when building AKMaj vocabulary
	if (in_vocab_type.compare("HKM") != 0
			&& in_vocab_type.compare("HKMAJ") != 0) {
		printf("-- Building nearest neighbors index\n");
		mytime = cv::getTickCount();

		((cv::Ptr<vlr::AKMajDB>) db)->buildNNIndex();

		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;
		printf("   Built in [%lf] ms\n", mytime);

		printf("-- Saving nearest neighbors index to [%s]\n",
				out_nn_index.c_str());
		mytime = cv::getTickCount();

		((cv::Ptr<vlr::AKMajDB>) db)->saveNNIndex(out_nn_index);

		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;
		printf("   Saved in [%lf] ms\n", mytime);
	}

	// Step 2/4: Quantize training data (several image descriptor matrices)
	printf("-- Creating vocabulary database with [%lu] images\n",
			descFilenames.size());
	db->clearDatabase();
	printf("   Clearing Inverted Files\n");

	cv::Mat imgDescriptors;
	int imgIdx = 0;

	for (std::string& keyFileName : descFilenames) {
		// Load descriptors
		FileUtils::loadDescriptors(keyFileName, imgDescriptors);

		// Check descriptors type
		// Note: for empty matrices FileStorage API sets as 0 the descriptor type
		// TODO Automatically identify if database uses a BoF model for binary or non-binary data
//		if (imgDescriptors.empty() == false
//				&& (imgDescriptors.type() == CV_8U) != isDescriptorBinary) {
//			fprintf(stderr,
//					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
//					isDescriptorBinary == true ? "binary" : "non-binary",
//					imgDescriptors.type() == CV_8U ? "binary" : "non-binary");
//			return EXIT_FAILURE;
//		}

		// Add image to database
		printf("   Adding image [%u] to database\n", imgIdx);
		try {
			db->addImageToDatabase(imgIdx, imgDescriptors);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}

		// Increase added images counter
		++imgIdx;
	}

	imgDescriptors.release();
	imgDescriptors = cv::Mat();

	CV_Assert(imgIdx >= 0 && (size_t ) imgIdx == descFilenames.size());

	printf("   Added [%u] images\n", imgIdx);

	// Step 3/4: Compute words weights and normalize DB

	vlr::WeightingType weighting = vlr::TF_IDF;

	if (in_weighting.compare("TF") == 0) {
		weighting = vlr::TF;
	} else if (in_weighting.compare("BIN") == 0) {
		weighting = vlr::BINARY;
	}

	printf("-- Computing words weights using a [%s] weighting scheme\n",
			weighting == vlr::TF_IDF ? "TF-IDF" : weighting == vlr::TF ? "TF" :
			weighting == vlr::BINARY ? "BINARY" : "UNKNOWN");

	db->computeWordsWeights(weighting);

	printf("-- Applying words weights to the database BoF vectors counts\n");
	db->createDatabase();

	vlr::NormType norm = vlr::NORM_L2;

	if (in_norm.compare("L1") == 0) {
		norm = vlr::NORM_L1;
	}

	printf("-- Normalizing database BoF vectors using [%s-norm]\n",
			norm == vlr::NORM_L1 ? "L1" :
			norm == vlr::NORM_L2 ? "L2" : "Unknown");
	db->normalizeDatabase(norm);

	printf("-- Saving inverted index to [%s]\n", out_inv_index.c_str());

	mytime = cv::getTickCount();
	db->saveInvertedIndex(out_inv_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Inverted index saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}

