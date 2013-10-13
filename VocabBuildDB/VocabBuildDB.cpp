/*
 * VocabBuildDB.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <stdlib.h>

int main(int argc, char **argv) {

	if (argc < 4 || argc > 7) {
		printf("Usage: %s <in.list> <in.tree> <out.tree> [use_tfidf:1] "
				"[normalize:1] [start_id:0] [distance_type:1]\n", argv[0]);
		return EXIT_FAILURE;
	}

	bool use_tfidf = true;
	bool normalize = true;

	char *list_in = argv[1];
	char *tree_in = argv[2];
	char *tree_out = argv[3];

	if (argc >= 5) {
		use_tfidf = atoi(argv[4]);
	}

	if (argc >= 6) {
		normalize = atoi(argv[5]);
	}

//    if (argc >= 7)
//	distance_type = (DistanceType) atoi(argv[7]);
//
//	switch (distance_type) {
//	case DistanceDot:
//		printf("[VocabMatch] Using distance Dot\n");
//		break;
//	case DistanceMin:
//		printf("[VocabMatch] Using distance Min\n");
//		break;
//	default:
//		printf("[VocabMatch] Using no known distance!\n");
//		break;
//	}

	cvflann::VocabTree tree;
	cv::Mat descriptors;

	// Step 2/4: Quantize training data (several image descriptor matrices)
	std::vector<cv::Mat> images;
	images.push_back(descriptors);
	printf("-- Creating vocabulary database with [%lu] images\n",
			images.size());
	tree.clearDatabase();
	printf("   Clearing Inverted Files\n");
	for (size_t imgIdx = 0; imgIdx < images.size(); imgIdx++) {
		printf("   Adding image [%lu] to database\n", imgIdx);
		tree.addImageToDatabase(imgIdx, images[imgIdx]);
	}

	// Step 3/4: Compute words weights and normalize DB
	const DBoW2::WeightingType weightingScheme = DBoW2::TF_IDF;
	printf("   Computing words weights using a [%s] weighting scheme\n",
			weightingScheme == DBoW2::TF_IDF ? "TF-IDF" :
			weightingScheme == DBoW2::TF ? "TF" :
			weightingScheme == DBoW2::IDF ? "IDF" :
			weightingScheme == DBoW2::BINARY ? "BINARY" : "UNKNOWN");
	tree.computeWordsWeights(descriptors.rows, weightingScheme);
	printf("   Applying words weights to the DB BoW vectors counts\n");
	tree.createDatabase();

	int normType = cv::NORM_L1;

	printf("   Normalizing DB BoW vectors using [%s]\n",
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");
	tree.normalizeDatabase(1, normType);

	std::string dbOut = "db.yaml.gz";
	printf("   Saving DB to [%s]\n", dbOut.c_str());
	tree.save(dbOut);



	return EXIT_SUCCESS;
}

