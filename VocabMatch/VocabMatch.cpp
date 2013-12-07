/*
 * VocabMatch.cpp
 *
 *  Created on: Oct 11, 2013
 *      Author: andresf
 */

#include <fstream>
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

	char *in_tree = argv[1];
	char* in_inverted_index = argv[2];
	char *in_db_desc_list = argv[3];
	char *in_queries_desc_list = argv[4];
	char *out_ranked_files_folder = argv[5];
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
	if (boost::regex_match(std::string(in_tree), DESCRIPTOR_REGEX) == false) {
		fprintf(stderr,
				"Input tree file must have the extension .yaml.gz or .xml.gz\n");
		return EXIT_FAILURE;
	}

	// Step 1/4: load tree + inverted index
	cv::Ptr<bfeat::VocabTreeBase> tree;

	if (is_binary == true) {
		tree = new bfeat::VocabTree<uchar, cv::Hamming>();
	} else {
		tree = new bfeat::VocabTree<float, cvflann::L2<float> >();
	}

	printf("-- Loading tree from [%s]\n", in_tree);

	mytime = cv::getTickCount();
	tree->load(std::string(in_tree));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Tree loaded in [%lf] ms, got [%lu] words \n", mytime,
			tree->size());

	printf("-- Loading inverted index [%s]\n", in_inverted_index);

	mytime = cv::getTickCount();
	tree->loadInvertedIndex(std::string(in_inverted_index));
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;

	printf("   Inverted index loaded in [%lf] ms\n", mytime);

	std::ifstream inputFileStream;
	std::string line;

	// Step 2/4: load names of database files
	printf("-- Loading names of database files\n");
	std::vector<std::string> db_desc_list;
	inputFileStream.open(in_db_desc_list, std::fstream::in);

	if (inputFileStream.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n",
				in_db_desc_list);
		return EXIT_FAILURE;
	}

	// Loading names of database files
	while (getline(inputFileStream, line)) {

		// Checking that filename refers to a compressed yaml or xml file
		if (boost::regex_match(line, DESCRIPTOR_REGEX) == false) {
			fprintf(stderr, "File [%s] must have the extension "
					".yaml.gz or .xml.gz\n", line.c_str());
			return EXIT_FAILURE;
		}

		db_desc_list.push_back(line);
	}
	// Close file
	inputFileStream.close();

	// Step 3/4: load list of queries descriptors
	printf("-- Loading list of queries descriptors\n");
	std::vector<std::string> query_filenames;
	inputFileStream.open(in_queries_desc_list, std::fstream::in);

	if (inputFileStream.is_open() == false) {
		fprintf(stderr, "Error opening file [%s] for reading\n",
				in_queries_desc_list);
		return EXIT_FAILURE;
	}

	// Loading file names in list into a vector
	while (getline(inputFileStream, line)) {

		// Checking that file exists, if not print error and exit
		struct stat buffer;
		if (stat(line.c_str(), &buffer) != 0) {
			fprintf(stderr, "File [%s] doesn't exist\n", line.c_str());
			return EXIT_FAILURE;
		}

		// Checking that line refers to a compressed yaml or xml file
		if (boost::regex_match(line, DESCRIPTOR_REGEX) == false) {
			fprintf(stderr, "File [%s] must have the extension "
					".yaml.gz or .xml.gz\n", line.c_str());
			return EXIT_FAILURE;
		}

		query_filenames.push_back(line);
	}
	// Close file
	inputFileStream.close();

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

	FILE *f_html = fopen(output_html, "w");
	HtmlResultsWriter::getInstance().writeHeader(f_html, top);
	if (f_html == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n", output_html);
		return EXIT_FAILURE;
	}

	for (size_t i = 0; i < query_filenames.size(); ++i) {
		// Initialize descriptors
		imgDescriptors = cv::Mat();

		// Load query descriptors
		FileUtils::loadDescriptors(query_filenames[i], imgDescriptors);

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
//			cv::Mat dbBowVector;
//			tree->getDbBowVector(j, dbBowVector);
//			std::cout << j << ") database BoW vector:\n" << dbBowVector << std::endl;
//			printf(
//					"   Match score between [%s] query image and [%lu] database image: %f\n",
//					query_filenames[i].c_str(), j, scores.at<float>(0, j));
//			getchar();
			printf(
					"   Match score between [%lu] query image and [%lu] database image: %f\n",
					i, j, scores.at<float>(0, j));
		}

		// Obtain indices of ordered scores
		cv::Mat perm;
		// Note: recall that the index of the images in the inverted file corresponds
		// to the zero-based line number in the file used to build the database.
		// Hence scores matrix and db_landmarks and db_filenames vectors.
		// are equally ordered.
		// Also the images in list_db and list_db_ld must be equally ordered,
		// that implies same number of elements.
		//
		// list_db      list_db_ld
		//   img1  --->  img1 ld1
		//   img2  --->  img2 ld1
		//   img3  --->  img3 ld1
		//   img4  --->  img4 ld2
		//   img5  --->  img5 ld2
		//   img6  --->  img6 ld2
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
		HtmlResultsWriter::getInstance().writeRow(f_html, query_filenames[i],
				scores, perm, top, db_desc_list);
		scores.release();
		perm.release();
	}

	HtmlResultsWriter::getInstance().writeFooter(f_html);
	fclose(f_html);

	return EXIT_SUCCESS;
}
