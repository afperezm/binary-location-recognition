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

	if (argc < 4 || argc == 5
			|| (argc > 5 && std::string(argv[4]).compare("-opts") != 0)) {
		printf(
				"\nUsage:\n"
						"\tVocabLearn <in.training.images.list> <in.vocab.type> <out.vocab> [-opts <key>=<value>]\n\n"
						"Vocabulary type:\n"
						"\tHKM: Hierarchical K-Means\n"
						"\tHKMAJ: Hierarchical K-Majority\n"
						"\tAKM: Approximate K-Means (Not yet supported)\n"
						"\tAKMAJ: Approximate K-Majority\n\n"
						"HKM and HKMAJ options:\n"
						"\tdepth=6\t\t\tbranch.factor=10\n"
						"\tmax.iterations=10\tcenters.init.method=RANDOM\n\n"
						"AKMAJ options:\n"
						"\tnum.clusters=1000000\t\tmax.iterations=10\n"
						"\tcenters.init.method=RANDOM\tnn.method=HIERARCHICAL\n"
						"\ttrees.number=4\t\t\ttrees.branch.factor=32\n"
						"\ttrees.max.leaf.size=100\t\ttrees.number.checks=4\n\n"
						"Centers initialization algorithms:\n"
						"\tRANDOM: in a random manner\n"
						"\tKMEANSPP: using k-means++ by Arthur and Vassilvitskii\n"
						"\tGONZALEZ: using Gonzalez algorithm\n\n"
						"Nearest Neighbors index type:\n"
						"\tLINEAR:\n"
						"\tHIERARCHICAL:\n\n"
				// Centers are spaced apart from each other
				);
		return EXIT_FAILURE;
	}

	std::string in_train_list = argv[1];
	std::string in_vocab_type = argv[2];
	std::string out_vocab = argv[3];

	if (out_vocab.substr(out_vocab.length() - 8, out_vocab.length()).compare(
			".yaml.gz") != 0) {
		fprintf(stderr,
				"Output file containing vocabulary must have the extension .yaml.gz\n");
		return EXIT_FAILURE;
	}

	cvflann::IndexParams vocabParams;
	cvflann::IndexParams nnIndexParams;
	if (in_vocab_type.compare("HKM") == 0
			|| in_vocab_type.compare("HKMAJ") == 0) {
		vocabParams = vlr::VocabTreeParams();
	} else if (in_vocab_type.compare("AKMAJ") == 0) {
		vocabParams = vlr::KMajorityParams();
		nnIndexParams = cvflann::HierarchicalClusteringIndexParams();
	} else {
		fprintf(stderr,
				"Invalid vocabulary type, choose among HKM, HKMAJ or AKMAJ\n");
		return EXIT_FAILURE;
	}

	// Generalize arguments to any vocabulary
	if (argc >= 5) {
		for (int var = 5; var < argc; ++var) {
			std::string arg = argv[var];
			size_t delimPos = arg.find("=");
			CV_Assert(delimPos != std::string::npos);
			std::string key = arg.substr(0, delimPos);
			std::string value = arg.substr(delimPos + 1, arg.length());
			if (key.compare("centers.init.method") == 0) {
				cvflann::flann_centers_init_t centersInitMethod =
						cvflann::FLANN_CENTERS_RANDOM;
				if (value.compare("KMEANSPP") == 0) {
					centersInitMethod = cvflann::FLANN_CENTERS_KMEANSPP;
				} else if (value.compare("GONZALEZ") == 0) {
					centersInitMethod = cvflann::FLANN_CENTERS_GONZALES;
				}
				vocabParams[key] = centersInitMethod;
			} else if (key.compare("nn.method") == 0) {
				vlr::indexType nnMethod = vlr::HIERARCHICAL;
				if (value.compare("LINEAR") == 0) {
					nnMethod = vlr::LINEAR;
				}
				vocabParams[key] = nnMethod;
			} else if (key.substr(0, 6).compare("trees.") == 0) {
				std::string nnIndexParam;
				if (key.compare("trees.number") == 0) {
					nnIndexParam = "trees";
				} else if (key.compare("trees.branch.factor") == 0) {
					nnIndexParam = "branching";
				} else if (key.compare("trees.max.leaf.size") == 0) {
					nnIndexParam = "leaf_size";
				} else if (key.compare("trees.number.checks") == 0) {
					nnIndexParam = "checks";
				}
				nnIndexParams[nnIndexParam] = atoi(value.c_str());
			} else {
				printf("%s --> %d\n", key.c_str(), atoi(value.c_str()));
				vocabParams[key] = atoi(value.c_str());
			}
		}
	}

	// Step 1: read list of descriptors files to build the vocabulary
	printf("-- Loading list of descriptors files\n");
	std::vector<std::string> descriptorsFilenames;
	FileUtils::loadList(in_train_list, descriptorsFilenames);
	printf("   Loaded, got [%lu] entries\n", descriptorsFilenames.size());

	// Step 2: setup data-set
	printf("-- Initializing dynamic descriptors matrix\n");
	vlr::Mat dataset(descriptorsFilenames);
	printf("   Initialized, got [%d] descriptors\n", dataset.rows);

	// Step 3: check vocabulary type and data type agree
	if (in_vocab_type.compare("HKM") == 0 ?
			dataset.type() != CV_32F : dataset.type() != CV_8U) {
		fprintf(stderr, "Vocabulary type does not coincide with data type\n");
		return EXIT_FAILURE;
	}

	// Step 4: build vocabulary
	cv::Ptr<vlr::VocabBase> vocab;
	if (in_vocab_type.compare("HKM") == 0) {
		vocab = new vlr::VocabTreeReal(dataset, vocabParams);
	} else if (in_vocab_type.compare("HKMAJ") == 0) {
		vocab = new vlr::VocabTreeBin(dataset, vocabParams);
	} else if (in_vocab_type.compare("AKMAJ") == 0) {
		vocab = new vlr::KMajority(dataset, vocabParams, nnIndexParams);
	}

	printf("-- Building [%s] vocabulary from [%d] feature vectors",
			in_vocab_type.c_str(), dataset.rows);
	for (cvflann::IndexParams::iterator it = vocabParams.begin();
			it != vocabParams.end(); ++it) {
		if (it->first.compare("centers.init.method") == 0) {
			cvflann::flann_centers_init_t centersInitMethod = it->second.cast<
					cvflann::flann_centers_init_t>();
			printf(", %s=%s", it->first.c_str(),
					centersInitMethod == cvflann::FLANN_CENTERS_RANDOM ?
							"RANDOM" :
					centersInitMethod == cvflann::FLANN_CENTERS_KMEANSPP ?
							"KMEANSPP" :
					centersInitMethod == cvflann::FLANN_CENTERS_GONZALES ?
							"GONZALEZ" : "UNKNOWN");
		} else if (it->first.compare("nn.method") == 0) {
			vlr::indexType nnMethod = it->second.cast<vlr::indexType>();
			printf(", %s=%s", it->first.c_str(),
					nnMethod == vlr::LINEAR ? "LINEAR" :
					nnMethod == vlr::HIERARCHICAL ? "HIERARCHICAL" : "UNKNOWN");
		} else {
			printf(", %s=%d", it->first.c_str(), it->second.cast<int>());
		}
	}
	for (cvflann::IndexParams::iterator it = nnIndexParams.begin();
			it != nnIndexParams.end(); ++it) {
		std::string nnIndexParam;
		if (it->first.compare("trees") == 0) {
			nnIndexParam = "trees.number";
		} else if (it->first.compare("branching") == 0) {
			nnIndexParam = "trees.branch.factor";
		} else if (it->first.compare("leaf_size") == 0) {
			nnIndexParam = "trees.max.leaf.size";
		} else if (it->first.compare("checks") == 0) {
			nnIndexParam = "trees.number.checks";
		}
		// Print only parameters with a defined conversion rule
		if (nnIndexParam.empty() == false) {
			printf(", %s=%d", nnIndexParam.c_str(), it->second.cast<int>());
		}
	}
	printf("\n");

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
