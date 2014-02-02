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

	if (argc < 6 || argc > 9) {
		printf("\nUsage:\n\tVocabBuildDB <in.db.descriptors.list> "
				"<in.tree> <out.inverted.index> <in.vocab.type>"
				" [in.weighting:TFIDF] [in.norm:L1]\n\n");
		// TODO Add options in usage text
		return EXIT_FAILURE;
	}

	std::string list_in = argv[1];
	std::string tree_in = argv[2];
	std::string out_inv_index = argv[3];
	std::string type = argv[4];

	std::string weighting = "TFIDF";
	bool normalize = false;
	std::string norm = "L1";

	if (argc >= 6) {
		weighting = argv[5];
	}

	if (argc >= 7) {
		normalize = true;
		norm = atoi(argv[6]);
	}

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(tree_in, expression) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	if (boost::regex_match(out_inv_index, expression) == false) {
		fprintf(stderr,
				"Output inverted index file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: read list of descriptors that shall be used to build the tree
	printf("-- Loading list of database images descriptors\n");
	std::vector<std::string> descFilenames;
	FileUtils::loadList(list_in, descFilenames);
	printf("   Loaded, got [%lu] entries\n", descFilenames.size());

	printf("-- Building database using [%lu] images\n", descFilenames.size());

	cv::Ptr<vlr::VocabDB> db;

	if (type.compare("HKM") == 0) {
		// HKM
		db = new vlr::HKMDB(false);
	} else if (type.compare("HKMAJ") == 0) {
		// HKMaj
		db = new vlr::HKMDB(true);
	} else {
		// AKMaj
		db = new vlr::AKMajDB();
	}

	printf("-- Reading tree from [%s]\n", tree_in.c_str());

	mytime = cv::getTickCount();
	db->loadBoFModel(tree_in);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			db->getNumOfWords());

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

	vlr::WeightingType weightingScheme = vlr::TF_IDF;

	if (weighting.compare("TF") == 0) {
		weightingScheme = vlr::TF;
	} else if (weighting.compare("BIN") == 0) {
		weightingScheme = vlr::BINARY;
	}

	printf("-- Computing words weights using a [%s] weighting scheme\n",
			weightingScheme == vlr::TF_IDF ? "TF-IDF" :
			weightingScheme == vlr::TF ? "TF" :
			weightingScheme == vlr::BINARY ? "BINARY" : "UNKNOWN");

	db->computeWordsWeights(weightingScheme);

	printf("-- Applying words weights to the database BoF vectors counts\n");
	db->createDatabase();

	int normType = cv::NORM_L1;

	if (norm.compare("L2") == 0) {
		normType = cv::NORM_L2;
	}

	if (normalize == true) {
		printf("-- Normalizing database BoF vectors using [%s]\n",
				normType == cv::NORM_L1 ? "L1-norm" :
				normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");
		db->normalizeDatabase(normType);
	}

	printf("-- Saving inverted index to [%s]\n", out_inv_index.c_str());

	mytime = cv::getTickCount();
	db->saveInvertedIndex(out_inv_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Inverted index saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}

