/*
 * VocabLearn.cpp
 *
 *  Created on: Oct 6, 2013
 *      Author: andresf
 */

#include <bitset>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <ctime>
#include <sys/stat.h>
#include <boost/regex.hpp>

#include <opencv2/core/internal.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/nonfree/nonfree.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <VocabBase.hpp>
#include <VocabTree.h>
#include <KMajority.h>

using std::vector;

double mytime;

int main(int argc, char **argv) {

	if (argc != 7) {
		printf(
				"\nUsage:\n"
						"\tVocabLearn <in.training.images.list> <in.vocab.type> <out.vocab>"
						" <in.depth> <in.branch.factor> <in.restarts>\n\n"
						"Vocabulary type:\n\n"
						"\tHKM: Hierarchical K-Means\n"
						"\tHKMAJ: Hierarchical K-Majority\n"
						"\tAKMAJ: Approximate K-Majority\n"
						"\tAKM: Approximate K-Means (Not yet supported)\n\n");
		return EXIT_FAILURE;
	}

	std::string in_train_list = argv[1];
	std::string in_vocab_type = argv[2];
	std::string out_vocab = argv[3];
	// TODO Generalize arguments to any vocabulary
	int in_depth = atoi(argv[4]);
	int in_branchFactor = atoi(argv[5]);
	int in_restarts = atoi(argv[6]);

	boost::regex expression("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

	if (boost::regex_match(out_vocab, expression) == false) {
		fprintf(stderr,
				"Output file containing vocabulary must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1: read list of descriptors files to build the vocabulary
	printf("-- Loading list of descriptors files\n");
	std::vector<std::string> descriptorsFilenames;
	FileUtils::loadList(in_train_list, descriptorsFilenames);
	printf("   Loaded, got [%lu] entries\n", descriptorsFilenames.size());

	// Step 3: build vocabulary
	printf("-- Initializing dynamic descriptors matrix\n");
	vlr::Mat dataset(descriptorsFilenames);
	printf("   Initialized, got [%d] descriptors\n", dataset.rows);

	cv::Ptr<vlr::VocabBase> vocab;

	vlr::VocabTreeParams params;
	params["branching"] = in_branchFactor;
	params["iterations"] = in_restarts;
	params["depth"] = in_depth;

	if (in_vocab_type.compare("HKM") == 0) {
		// Cluster descriptors using HKM
		vocab = new vlr::VocabTreeReal(dataset, params);
	} else if (in_vocab_type.compare("HKMAJ") == 0) {
		// Cluster descriptors using HKMaj
		vocab = new vlr::VocabTreeBin(dataset, params);
	} else if (in_vocab_type.compare("AKMAJ") == 0) {
		// Cluster descriptors using AKMaj
		vocab = new vlr::KMajority(in_branchFactor, in_restarts, dataset);
	} else {
		fprintf(stderr,
				"Invalid vocabulary type, choose among HKM, HKMAJ or AKMAJ\n");
		return EXIT_FAILURE;
	}

	if (in_vocab_type.compare("HKM") == 0
			|| in_vocab_type.compare("HKMAJ") == 0) {
		printf(
				"-- Building [%s] vocabulary from [%d] feature vectors, branch factor [%d], max iterations [%d], depth [%d], centers initialization algorithm [%s]\n",
				in_vocab_type.c_str(), dataset.rows,
				params["branching"].cast<int>(),
				params["iterations"].cast<int>(), params["depth"].cast<int>(),
				params["centers_init"].cast<cvflann::flann_centers_init_t>()
						== cvflann::FLANN_CENTERS_RANDOM ? "random" :
				params["centers_init"].cast<cvflann::flann_centers_init_t>()
						== cvflann::FLANN_CENTERS_GONZALES ? "gonzalez" :
				params["centers_init"].cast<cvflann::flann_centers_init_t>()
						== cvflann::FLANN_CENTERS_KMEANSPP ?
						"k-means++" : "unknown");

		if (in_vocab_type.compare("HKM") == 0 && dataset.type() == CV_8U) {
			fprintf(stderr,
					"Vocabulary type [%s] doesn't match data set type [%s]\n",
					in_vocab_type.c_str(),
					dataset.type() == CV_8U ? "binary" : "non-binary");
			return EXIT_FAILURE;
		}

	} else {
		printf(
				"-- Building [%s] vocabulary from [%d] feature vectors, branch factor [%d], max iterations [%d]\n",
				in_vocab_type.c_str(), dataset.rows, in_branchFactor,
				in_restarts);
	}

	mytime = cv::getTickCount();
	vocab->build();
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf(
			"   Vocabulary created from [%d] descriptors in [%lf] ms with [%lu] words\n",
			dataset.rows, mytime, vocab->size());

	printf("-- Saving vocabulary to [%s]\n", out_vocab.c_str());

	mytime = cv::getTickCount();
	vocab->save(out_vocab);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Vocabulary saved in [%lf] ms\n", mytime);

	return EXIT_SUCCESS;
}
