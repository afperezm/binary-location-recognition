/*
 * GeomVerify.cpp
 *
 *  Created on: Nov 20, 2013
 *      Author: andresf
 */

#include <iostream>

#include <boost/regex.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/flann/logger.h>
#include <opencv2/highgui/highgui.hpp>

#include <matching.hpp>

#include <FileUtils.hpp>
#include <FunctionUtils.hpp>
#include <HtmlResultsWriter.hpp>
#include <VocabTree.h>

double mytime;

// For each query
//	 - Load its keys
//	 - Load the list of its ranked candidates
//	 - For each ranked candidate
//		 - Load its keys
//		 - Find correspondences between query and candidate by intersecting direct index results
//		 - Match together all keypoints in same node and maybe apply some criteria for matching results
//		 - Compute a projective transformation between query and ranked file using direct index
//		 - Obtain number of inliers
//	 - Re-order list of candidates by its number of inliers

int main(int argc, char **argv) {

	if (argc < 11 || argc > 16) {
		printf(
				"\nUsage:\n"
						"\tGeomVerify <in.tree> <in.direct.index> "
						"<in.ranked.files.folder> <in.ranked.files.prefix> "
						"<in.db.descriptors.list> <in.db.keypoints.folder> <in.queries.descriptors.list> <in.queries.keypoints.folder> "
						"<out.re-ranked.files.folder> <in.top.candidates> "
						"[in.type.binary:1] [in.topKeypoints:500] [in.ratio.thr:0.8] [im.min.matches:8] [in.ransac.thr:10]"
						"\n\n");
		return EXIT_FAILURE;
	}

	std::string in_tree = argv[1];
	std::string in_direct_index = argv[2];
	std::string in_ranked_lists_folder = argv[3];
	std::string in_prefix = argv[4];
	std::string in_db_desc_list = argv[5]; // Used to obtain the mapping dbimg.candidate -> dbimg.id
	std::string in_db_keys_folder = argv[6]; // Used to load query ranked candidates
	std::string in_queries_desc_list = argv[7]; // Quantized in tree to match keypoints represented by the same word
	std::string in_queries_keys_folder = argv[8];
	std::string out_ranked_lists_folder = argv[9];

	int topCandidates = atoi(argv[10]);
	bool isBinary = argc >= 12 ? atoi(argv[11]) : true;
	int topKeypoints = argc >= 13 ? atoi(argv[12]) : 500;
	double ratioThreshold = argc >= 14 ? atof(argv[13]) : 0.8;
	int ransacMinMatches = argc >= 15 ? atoi(argv[14]) : 8;
	double ransacThreshold = argc >= 16 ? atof(argv[15]) : 10.0;

	// Step 1/4: load tree + direct index
	cv::Ptr<bfeat::VocabTreeBase> tree;
	cv::Ptr<bfeat::DirectIndex> directIndexQueries = new bfeat::DirectIndex();
	cv::Ptr<bfeat::DirectIndex> directIndexCandidates =
			new bfeat::DirectIndex();

	if (isBinary == true) {
		tree = new bfeat::VocabTreeBin();
	} else {
		tree = new bfeat::VocabTreeReal();
	}

	printf(
			"-- Running spatial verification using topCandidates=[%d] topKeypoints=[%d] "
					"ratioThr=[%2.1f] ransacMinMatches=[%d] ransacThr=[%2.1f]\n",
			topCandidates, topKeypoints, ratioThreshold, ransacMinMatches,
			ransacThreshold);

	printf("-- Loading tree from [%s]\n", in_tree.c_str());
	mytime = cv::getTickCount();
	tree->load(in_tree);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Loaded in [%lf] ms, got [%lu] words \n", mytime, tree->size());

	printf("-- Loading direct index [%s]\n", in_direct_index.c_str());
	mytime = cv::getTickCount();
	directIndexCandidates->load(in_direct_index);
	mytime = ((double) cv::getTickCount() - mytime) / cv::getTickFrequency()
			* 1000;
	printf("   Loaded in [%lf] ms, got [%lu] images\n", mytime,
			directIndexCandidates->size());

	// Step 2a: load list of queries descriptors
	printf("-- Loading list of queries descriptors\n");
	std::vector<std::string> queries_desc_list;
	FileUtils::loadList(in_queries_desc_list, queries_desc_list);
	printf("   Loaded, got [%lu] entries\n", queries_desc_list.size());

	// Step 2b: load queries descriptors

	uint wordIdx; // unused
	double wordWeight; // unused
	int nodeAtL, imgIdx = 0;
	cv::Mat queryDescriptors;

	tree->setDirectIndexLevel(
			tree->getDepth() - 1 - directIndexCandidates->getLevel());

	printf(
			"-- Loading queries descriptors and adding them to the direct index, level=[%d]\n",
			tree->getDirectIndexLevel());
	// Loop over list of queries descriptors
	for (std::string descriptorsFilename : queries_desc_list) {
		// Load query descriptors
		FileUtils::loadDescriptors(descriptorsFilename, queryDescriptors);
		// Loop over query descriptors
		for (int i = 0; i < queryDescriptors.rows; ++i) {
			// Propagate descriptor down tree
			tree->quantize(queryDescriptors.row(i), wordIdx, wordWeight,
					nodeAtL);
			// Add it to a new image to the direct index
			directIndexQueries->addFeature(imgIdx, nodeAtL, i);
		}
		++imgIdx;
	}
	printf("   Loaded and added descriptors of [%lu] images\n",
			directIndexCandidates->size());

	// Step 3a: load list of queries key-points

	// Step 3b: load names of database files
	printf("-- Loading names of database files\n");
	std::vector<std::string> db_desc_list;
	FileUtils::loadList(in_db_desc_list, db_desc_list);
	printf("   Loaded, got [%lu] entries\n", db_desc_list.size());

	// Step 4/4: load and process queries key-points
	printf("-- Loading and processing queries key-points\n");
	std::vector<cv::KeyPoint> queryKeypoints;

	std::vector<std::string> ranked_candidates_list,
			geom_ranked_candidates_list;
	std::vector<cv::KeyPoint> candidateKeypoints;
	std::stringstream ranked_list_fname;

	std::vector<cv::DMatch> matchesCandidateToQuery, inlierMatches;
	std::vector<cv::Point2f> matchedCandidatePoints, matchedQueryPoints;
	int candidateImgId, queryImgId, top = -1;

	cv::Mat inliers_idx, candidates_inliers, candidates_inliers_idx, H;

	cv::Mat imgOut;
	cv::Mat queryImg, candidateImg;
	cv::Mat candidateDescriptors;

	std::string queryBase, candidateBase;

	cvflann::Logger::setDestination("inliers.log");

	HtmlResultsWriter::getInstance().open("results_gv.html", topCandidates);

	// Loop over list of queries key-points
	for (size_t i = 0; i < queries_desc_list.size(); ++i) {
		printf("-- Processing query [%lu] - [%s]\n", i,
				queries_desc_list[i].c_str());
		queryImgId = i;

		queryBase = queries_desc_list[i].substr(8,
				queries_desc_list[i].length() - 16);

		// Step 4a: load and pre-process query features
		printf("   Load and pre-process query features\n");
		FileUtils::loadKeypoints(
				in_queries_keys_folder + "/" + queryBase + "_kpt.yaml.gz",
				queryKeypoints);
		FileUtils::loadDescriptors(queries_desc_list[i], queryDescriptors);
		filterFeatures(queryKeypoints, queryDescriptors, topKeypoints);

		// Step 4b: load list of query ranked candidates
		printf("   Loading list of ranked candidates\n");
		// Load list of query ranked candidates
		// Note: recall that elements in the lists of queries key-points and descriptors
		// follow the same order and hence using query key-points filename position to build
		// its ranked candidates filename its legal
		ranked_list_fname.str("");
		ranked_list_fname << in_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		FileUtils::loadList(ranked_list_fname.str(), ranked_candidates_list);
		printf("   Loaded, got [%lu] candidates\n",
				ranked_candidates_list.size());

		top = MIN(int(ranked_candidates_list.size()), topCandidates);

		candidates_inliers = cv::Mat::zeros(1, top, cv::DataType<int>::type);

		queryImg = cv::imread("oxbuild_images/" + queryBase + ".jpg",
				CV_LOAD_IMAGE_GRAYSCALE);

//		imgOut = cv::Mat();
//		cv::drawKeypoints(queryImg, queryKeypoints, imgOut,
//				cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DEFAULT);
//		cv::namedWindow("oxbuild_images/" + queryBase + ".jpg",
//				cv::WINDOW_NORMAL);
//		cv::resizeWindow("oxbuild_images/" + queryBase + ".jpg", 500, 500);
//		cv::imshow("oxbuild_images/" + queryBase + ".jpg", imgOut);
//		cv::waitKey(0);

		// Step 4c: load and pre-process candidate features
		for (size_t j = 0; int(j) < top; ++j) {

			printf("   Load and pre-process candidate features\n");
			FileUtils::loadKeypoints(
					in_db_keys_folder + "/" + ranked_candidates_list[j]
							+ "_kpt.yaml.gz", candidateKeypoints);
			FileUtils::loadDescriptors(
					"db/" + ranked_candidates_list[j] + ".yaml.gz",
					candidateDescriptors);
			filterFeatures(candidateKeypoints, candidateDescriptors,
					topKeypoints);

			// Searching putative matches
			printf("   Matching key-points of query [%lu] "
					"against candidate [%lu]\n", i, j);

			// Id of database image
			std::vector<std::string>::iterator it = std::find(
					db_desc_list.begin(), db_desc_list.end(),
					"db/" + ranked_candidates_list[j] + ".yaml.gz");

			if (it == db_desc_list.end()) {
				throw std::runtime_error("Candidate [%s] not found "
						"in list of database filenames");
			}

			candidateImgId = std::distance(db_desc_list.begin(), it);

			candidateBase = ranked_candidates_list[j];
			candidateImg = cv::imread(
					"oxbuild_images/" + candidateBase + ".jpg",
					CV_LOAD_IMAGE_GRAYSCALE);

//			imgOut = cv::Mat();
//			cv::drawKeypoints(candidateImg, candidateKeypoints, imgOut,
//					cv::Scalar(255, 255, 0), cv::DrawMatchesFlags::DEFAULT);
//			cv::namedWindow("oxbuild_images/" + candidateBase + ".jpg",
//					cv::WINDOW_NORMAL);
//			cv::resizeWindow("oxbuild_images/" + candidateBase + ".jpg", 500,
//					500);
//			cv::imshow("oxbuild_images/" + candidateBase + ".jpg", imgOut);
//			cv::waitKey(0);

			mytime = cv::getTickCount();

			// TODO Use the direct index to pre-filter query and candidate key-points

			matchKeypoints(candidateKeypoints, candidateDescriptors,
					queryKeypoints, queryDescriptors, matchesCandidateToQuery,
					topKeypoints, ratioThreshold);

			matchedCandidatePoints.clear();
			matchedQueryPoints.clear();

			for (cv::DMatch& match : matchesCandidateToQuery) {
//				double kptsDist = cv::norm(
//						cv::Point(candidateKeypoints[match.queryIdx].pt.x,
//								candidateKeypoints[match.queryIdx].pt.y)
//								- cv::Point(queryKeypoints[match.trainIdx].pt.x,
//										queryKeypoints[match.trainIdx].pt.y));
//
//				// Apply a proximity threshold
//				if (kptsDist > 200) {
//					continue;
//				}
				// Add points to vectors of matched
				matchedQueryPoints.push_back(queryKeypoints[match.trainIdx].pt);
				matchedCandidatePoints.push_back(
						candidateKeypoints[match.queryIdx].pt);
			}

			mytime = (double(cv::getTickCount()) - mytime)
					/ cv::getTickFrequency() * 1000;

			printf("   Found [%d] putative matches in [%lf] ms\n",
					int(matchesCandidateToQuery.size()), mytime);

//			imgOut = cv::Mat();
//			cv::drawMatches(candidateImg, candidateKeypoints, queryImg,
//					queryKeypoints, matchesCandidateToQuery, imgOut);
//			cv::namedWindow(
//					"oxbuild_images/" + queryBase + "_" + candidateBase
//							+ ".jpg", cv::WINDOW_NORMAL);
//			cv::resizeWindow(
//					"oxbuild_images/" + queryBase + "_" + candidateBase
//							+ ".jpg", 500, 500);
//			cv::imshow(
//					"oxbuild_images/" + queryBase + "_" + candidateBase
//							+ ".jpg", imgOut);
//			cv::waitKey(0);

			if ((int(matchesCandidateToQuery.size())) < ransacMinMatches) {
				fprintf(stderr, "   Cannot compute homography, "
						"at least [%d] putative matches are needed\n",
						ransacMinMatches);
			} else {
				// Compute a projective transformation between query and ranked file using direct index
				printf("   Computing projective transformation "
						"between query [%lu] and candidate [%lu]\n", i, j);

				inliers_idx = cv::Mat();

				mytime = cv::getTickCount();
				H = cv::findHomography(matchedCandidatePoints,
						matchedQueryPoints, CV_RANSAC, ransacThreshold,
						inliers_idx);
				mytime = (double(cv::getTickCount()) - mytime)
						/ cv::getTickFrequency() * 1000;

				// Obtain number of inliers
				int numInliers = int(sum(inliers_idx)[0]);

				printf(
						"   Computed homography in [%0.3fs], found [%d] inliers\n",
						mytime, numInliers);

				candidates_inliers.at<int>(0, j) = numInliers;

				inlierMatches.clear();
				for (int i = 0; i < inliers_idx.rows; ++i) {
					if (int(inliers_idx.at<uchar>(i)) == int(1)) {
						inlierMatches.push_back(matchesCandidateToQuery.at(i));
					}
				}
				imgOut = cv::Mat();
				cv::drawMatches(candidateImg, candidateKeypoints, queryImg,
						queryKeypoints, inlierMatches, imgOut);
				cv::imwrite(
						out_ranked_lists_folder + "/match_" + queryBase + "_"
								+ candidateBase + ".jpg", imgOut);
//				cv::namedWindow(
//						"oxbuild_images/match_" + queryBase + "_"
//								+ candidateBase + ".jpg", cv::WINDOW_NORMAL);
//				cv::resizeWindow(
//						"oxbuild_images/match_" + queryBase + "_"
//								+ candidateBase + ".jpg", 500, 500);
//				cv::imshow(
//						"oxbuild_images/match_" + queryBase + "_"
//								+ candidateBase + ".jpg", imgOut);
//				cv::waitKey(0);

			}

			cvflann::Logger::log(0,
					"query=[%s] candidate=[%s] numberInliers=[%d]\n",
					queryBase.c_str(), candidateBase.c_str(),
					candidates_inliers.at<int>(0, j));

		}

		printf("-- Re-ranking candidates list\n");

		// Re-order list of candidates by its inlier number
		sortIdx(candidates_inliers, candidates_inliers_idx, CV_SORT_DESCENDING);

		// Copying re-ranked candidates
		geom_ranked_candidates_list.clear();
		for (size_t j = 0; int(j) < top; ++j) {
			geom_ranked_candidates_list.push_back(
					ranked_candidates_list[candidates_inliers_idx.at<int>(0, j)]);
		}

#if GVVERBOSE
		printf("Original ranked candidates list:\n");
		for (std::string candidate : ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

#if GVVERBOSE
		printf("Re-ranked candidates list:\n");
		for (std::string candidate : geom_ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

		// Copying non re-ranked candidates
		geom_ranked_candidates_list.insert(geom_ranked_candidates_list.end(),
				ranked_candidates_list.begin() + top,
				ranked_candidates_list.end());

		printf("   Done, re-ranked top [%d] candidates out of [%lu]\n", top,
				geom_ranked_candidates_list.size());

#if GVVERBOSE
		printf("Full re-ranked candidates list:\n");
		for (std::string candidate : geom_ranked_candidates_list) {
			printf("%s, ", candidate.c_str());
		}
		printf("\n");
#endif

		printf("-- Saving list of re-ranked candidates\n");

		ranked_list_fname.str("");
		ranked_list_fname << out_ranked_lists_folder << "/query_" << i
				<< "_ranked.txt";
		FileUtils::saveList(ranked_list_fname.str(),
				geom_ranked_candidates_list);

		printf("   Done, saved [%lu] entries\n",
				geom_ranked_candidates_list.size());
	}

	HtmlResultsWriter::getInstance().close();

}
