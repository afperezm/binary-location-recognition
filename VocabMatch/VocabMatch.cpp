/*
 * VocabMatch.cpp
 *
 *  Created on: Oct 11, 2013
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

#include <FunctionUtils.hpp>
#include <HtmlResultsWriter.hpp>

double mytime;

const static boost::regex DESCRIPTOR_REGEX("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

/**
 * Filters a set of features by keeping only those inside the region determined by query.
 *
 * @param query
 * @param descriptors
 *
 * @return number of filtered out features
 */
int filterFeaturesByRegion(FileUtils::Query& query, cv::Mat& descriptors);

int main(int argc, char **argv) {

	if (argc < 6 || argc > 12) {
		printf(
				"\nUsage:\n\t"
						"VocabMatch <in.vocab> <in.inverted.index> <in.db.desc.list> <in.queries.list>"
						" <out.ranked.files.folder> [in.num.neighbors:ALL] [in.norm:L2] [in.scoring:COS] [out.results:results.html]"
						" [in.use.regions:0] [in.nn.index]\n\n"
						"Norm:\n"
						"\tL1: L1-norm\n"
						"\tL2: L2-norm\n\n"
						"Distance:\n"
						"\tL1: Manhattan distance or Sum of absolute differences\n"
						"\tL2: Euclidean distance or Sum of squared differences\n"
						"\tCOS: Cosine distance or Euclidean dot product\n\n");
		return EXIT_FAILURE;
	}

	std::string in_vocab = argv[1];
	std::string in_inverted_index = argv[2];
	std::string in_db_desc_list = argv[3];
	std::string in_queries_desc_list = argv[4];
	std::string out_ranked_files_folder = argv[5];
	int in_num_nbrs = -1;
	std::string in_norm = "L2";
	std::string in_scoring = "COS";
	std::string out_html = "results.html";
	bool in_use_regions = false;
	std::string in_nn_index = "nn_index.bin";

	if (argc >= 7) {
		in_num_nbrs = atoi(argv[6]);
	}

	if (argc >= 8) {
		in_norm = argv[7];
	}

	if (argc >= 9) {
		in_scoring = argv[8];
	}

	if (argc >= 10) {
		out_html = argv[9];
	}

	if (argc >= 11) {
		in_use_regions = atoi(argv[10]);
	}

	if (argc >= 12) {
		in_nn_index = argv[11];
	}

	// Checking that database filename refers to a compressed YAML or XML file
	if (boost::regex_match(in_vocab, DESCRIPTOR_REGEX) == false) {
		fprintf(stderr,
				"Input vocabulary file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load vocabulary + inverted index
	cv::Ptr<vlr::VocabDB> db;

	std::string in_type = vlr::VocabBase::loadVocabType(in_vocab);

	if (in_type.compare("HKM") == 0) {
		// Instantiate a DB supported by a HKM vocabulary
		db = new vlr::HKMDB(false);
	} else if (in_type.compare("HKMAJ") == 0) {
		// Instantiate a DB supported by a HKMaj vocabulary
		db = new vlr::HKMDB(true);
	} else {
		// Instantiate a DB supported by a AKMaj vocabulary
		db = new vlr::AKMajDB();
	}

	printf("-- Loading vocabulary from [%s]\n", in_vocab.c_str());

	mytime = cv::getTickCount();
	db->loadBoFModel(in_vocab);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Vocabulary loaded in [%lf] ms, got [%lu] words \n", mytime,
			db->getNumOfWords());

	// Load nearest neighbor index when scoring using an AKMaj vocabulary
	if (in_type.compare("HKM") != 0 && in_type.compare("HKMAJ") != 0) {

		printf("-- Loading nearest neighbors index from [%s]\n",
				in_nn_index.c_str());

		mytime = cv::getTickCount();
		((cv::Ptr<vlr::AKMajDB>) db)->loadNNIndex(in_nn_index);
		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;

		printf("   Loaded in [%lf] ms\n", mytime);
	}

	printf("-- Loading inverted index [%s]\n", in_inverted_index.c_str());

	mytime = cv::getTickCount();
	db->loadInvertedIndex(in_inverted_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Inverted index loaded in [%lf] ms\n", mytime);

	// Step 2/4: load names of database files
	printf("-- Loading names of database files\n");
	std::vector<std::string> db_desc_list;
	FileUtils::loadList(in_db_desc_list, db_desc_list);
	printf("   Loaded, got [%lu] entries\n", db_desc_list.size());

	// Step 3/4: load list of queries descriptors
	printf("-- Loading list of queries descriptors\n");
	std::vector<FileUtils::Query> query_filenames;
	FileUtils::loadQueriesList(in_queries_desc_list, query_filenames);
	printf("   Loaded, got [%lu] entries\n", query_filenames.size());

	// Step 4/4: score each query
	vlr::NormType norm = vlr::NORM_L1;

	if (in_norm.compare("L2") == 0) {
		norm = vlr::NORM_L2;
	}

	vlr::ScoringType distance = vlr::COS;

	if (in_scoring.compare("L1") == 0) {
		distance = vlr::L1;
	} else if (in_scoring.compare("L2") == 0) {
		distance = vlr::L2;
	}

	printf(
			"-- Scoring [%lu] query images against [%d] database images using [%s-norm] and [%s distance]\n",
			query_filenames.size(), db->getInvertedIndex()->m_numDbImages,
			norm == vlr::NORM_L1 ? "L1" :
			norm == vlr::NORM_L2 ? "L2" : "Unknown",
			distance == vlr::L1 ? "L1" : distance == vlr::L2 ? "L2" :
			distance == vlr::COS ? "Cosine" : "Unknown");

	cv::Mat imgDescriptors;
	cv::Mat scores;

	// Compute the number of candidates
	int top =
			in_num_nbrs != -1 ?
					in_num_nbrs : db->getInvertedIndex()->m_numDbImages;

	HtmlResultsWriter::getInstance().open(out_html, top);

	for (size_t i = 0; i < query_filenames.size(); ++i) {
		// Initialize descriptors
		imgDescriptors = cv::Mat();

		// Load query descriptors
		FileUtils::loadDescriptors(query_filenames[i].name, imgDescriptors);

		if (in_use_regions == true) {
			// Load key-points and use them to filter the features

			int numFilteredFeatures = filterFeaturesByRegion(query_filenames[i],
					imgDescriptors);

			printf("   Filtered out [%d] features\n", numFilteredFeatures);

		}

		if (imgDescriptors.empty() == true) {
			continue;
		}

		// Check type of descriptors
		bool is_binary = in_type.compare("HKM") != 0;
		if ((imgDescriptors.type() == CV_8U) != is_binary) {
			fprintf(stderr,
					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
					is_binary == true ? "binary" : "non-binary",
					imgDescriptors.type() == CV_8U ? "binary" : "real");
			return EXIT_FAILURE;
		}

		// Score query BoF vector against database images BoF vectors
		mytime = cv::getTickCount();
		try {
			db->scoreQuery(imgDescriptors, scores, norm, distance);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}
		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;
		imgDescriptors.release();

		// Print to standard output the matching scores between
		// the query BoF vector and the database images BoF vectors
		for (size_t j = 0; (int) j < scores.cols; ++j) {
			printf(
					"   Match score between [%lu] query image and [%lu] database image: %f\n",
					i, j, scores.at<float>(0, j));
		}

		// Obtain indices of ordered scores
		cv::Mat perm;
		// Note: recall that the index of the images in the inverted file corresponds
		// to the zero-based line number in the file used to build the database.
		cv::sortIdx(scores, perm, cv::SORT_EVERY_ROW + cv::SORT_DESCENDING);

		std::stringstream ranked_list_fname;
		printf("%lu) %s\n", i, query_filenames[i].name.c_str());
		ranked_list_fname << out_ranked_files_folder << "/query_" << i
				<< "_ranked.txt";

		std::ofstream f_ranked_list(ranked_list_fname.str().c_str(),
				std::fstream::out);
		if (f_ranked_list.good() == false) {
			fprintf(stderr, "Error opening file [%s] for writing\n",
					ranked_list_fname.str().c_str());
			return EXIT_FAILURE;
		}
		for (int j = 0; j < top; ++j) {
			// Get base filename: remove extension and folder path
			std::string d_base = FunctionUtils::basify(
					db_desc_list[perm.at<int>(0, j)]);
			f_ranked_list << d_base + "\n";
		}
		f_ranked_list.close();

		// Print to a file the ranked list of candidates ordered by score in HTML format
		HtmlResultsWriter::getInstance().writeRow(query_filenames[i].name,
				scores, perm, top, db_desc_list);
		scores.release();
		perm.release();
	}

	HtmlResultsWriter::getInstance().close();

	return EXIT_SUCCESS;
}

int filterFeaturesByRegion(FileUtils::Query& q, cv::Mat& descriptors) {

	cv::Mat filteredDescriptors;
	std::vector<cv::KeyPoint> keypoints, filteredKeypoints;
	std::string queries_keys_folder = "queries", queryBase;

	queryBase = q.name.substr(8, q.name.length() - 16);

	FileUtils::loadKeypoints(
			queries_keys_folder + "/" + queryBase + "_kpt.yaml.gz", keypoints);

	for (int i = 0; i < descriptors.rows; ++i) {
		if (keypoints[i].pt.x > q.x1 && keypoints[i].pt.x < q.x2
				&& keypoints[i].pt.y > q.y1 && keypoints[i].pt.y < q.y2) {
			filteredKeypoints.push_back(keypoints[i]);
			filteredDescriptors.push_back(descriptors.row(i));
		}
	}

	int numFilteredFeatures = descriptors.rows - filteredDescriptors.rows;

	keypoints.clear();
	keypoints = filteredKeypoints;

	descriptors.release();
	descriptors = filteredDescriptors;

	CV_Assert(descriptors.rows == int(keypoints.size()));

	return numFilteredFeatures;

}
