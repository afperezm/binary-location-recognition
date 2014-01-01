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

#include <FileUtils.hpp>

#include <FunctionUtils.hpp>
#include <HtmlResultsWriter.hpp>

double mytime;

const static boost::regex DESCRIPTOR_REGEX("^(.+)(\\.)(yaml|xml)(\\.)(gz)$");

int main(int argc, char **argv) {

	if (argc < 7 || argc > 9) {
		printf(
				"\nUsage:\n\t"
						"VocabMatch <in.tree> <in.inverted.index> <in.db.desc.list> <in.queries.desc.list> <out.ranked.files.folder>"
						" <in.num.neighbors> [in.type.binary:1] [out.results:results.html]\n\n");
		return EXIT_FAILURE;
	}

	std::string in_tree = argv[1];
	std::string in_inverted_index = argv[2];
	std::string in_db_desc_list = argv[3];
	std::string in_queries_desc_list = argv[4];
	std::string out_ranked_files_folder = argv[5];
	uint num_nbrs = atoi(argv[6]);
	bool is_binary = true;
	const char *output_html = const_cast<char*>("results.html");

	if (argc >= 8) {
		is_binary = atoi(argv[7]);
	}

	if (argc >= 9) {
		output_html = argv[8];
	}

	// Checking that database filename refers to a compressed yaml or xml file
	if (boost::regex_match(in_tree, DESCRIPTOR_REGEX) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load tree + inverted index
	cv::Ptr<bfeat::VocabTreeBase> tree;

	if (is_binary == true) {
		tree = new bfeat::VocabTreeBin();
	} else {
		tree = new bfeat::VocabTreeReal();
	}

	printf("-- Loading tree from [%s]\n", in_tree.c_str());

	mytime = cv::getTickCount();
	tree->load(in_tree);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	printf("-- Loading inverted index [%s]\n", in_inverted_index.c_str());

	mytime = cv::getTickCount();
	tree->loadInvertedIndex(in_inverted_index);
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
	std::vector<std::string> query_filenames;
	FileUtils::loadList(in_queries_desc_list, query_filenames);
	printf("   Loaded, got [%lu] entries\n", query_filenames.size());

	// Step 4/4: score each query
	int normType = cv::NORM_L1;

	printf(
			"-- Scoring [%lu] query images against [%lu] database images using [%s]\n",
			query_filenames.size(), db_desc_list.size(),
			normType == cv::NORM_L1 ? "L1-norm" :
			normType == cv::NORM_L2 ? "L2-norm" : "UNKNOWN-norm");

	cv::Mat imgDescriptors;
	cv::Mat scores;

	// Compute the number of candidates
	int top = MIN (num_nbrs, db_desc_list.size());

	HtmlResultsWriter::getInstance().open(output_html, top);

	for (size_t i = 0; i < query_filenames.size(); ++i) {
		// Initialize descriptors
		imgDescriptors = cv::Mat();

		// Load query descriptors
		FileUtils::loadDescriptors(query_filenames[i], imgDescriptors);

		if (imgDescriptors.empty() == true) {
			continue;
		}

		// Check type of descriptors
		if ((imgDescriptors.type() == CV_8U) != is_binary) {
			fprintf(stderr,
					"Descriptor type doesn't coincide, it is said to be [%s] while it is [%s]\n",
					is_binary == true ? "binary" : "non-binary",
					imgDescriptors.type() == CV_8U ? "binary" : "real");
			return EXIT_FAILURE;
		}

		// Score query bow vector against database images bow vectors
		mytime = cv::getTickCount();
		try {
			tree->scoreQuery(imgDescriptors, scores, cv::NORM_L1);
		} catch (const std::runtime_error& error) {
			fprintf(stderr, "%s\n", error.what());
			return EXIT_FAILURE;
		}
		mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
				* 1000;
		imgDescriptors.release();

		// Print to standard output the matching scores between
		// the query bow vector and the database images bow vectors
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
		printf("%lu) %s\n", i, query_filenames[i].c_str());
		ranked_list_fname << out_ranked_files_folder << "/query_" << i
				<< "_ranked.txt";

		FILE *f_ranked_list = fopen(ranked_list_fname.str().c_str(), "w");
		if (f_ranked_list == NULL) {
			fprintf(stderr, "Error opening file [%s] for writing\n",
					ranked_list_fname.str().c_str());
			return EXIT_FAILURE;
		}
		for (int j = 0; j < top; ++j) {
			std::string d_base = db_desc_list[perm.at<int>(0, j)];
			fprintf(f_ranked_list, "%s\n",
					d_base.substr(0, d_base.size() - 8).substr(3).c_str());
		}
		fclose(f_ranked_list);

		// Print to a file the ranked list of candidates ordered by score in HTML format
		HtmlResultsWriter::getInstance().writeRow(query_filenames[i], scores,
				perm, top, db_desc_list);
		scores.release();
		perm.release();
	}

	HtmlResultsWriter::getInstance().close();

	return EXIT_SUCCESS;
}
