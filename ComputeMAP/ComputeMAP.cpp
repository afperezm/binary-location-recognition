/*
 * ComputeMAP.cpp
 *
 *  Created on: Nov 1, 2013
 *      Author: andresf
 */

#include <fstream>
#include <set>
#include <string>
#include <vector>
#include <stdlib.h>

#include <FileUtils.hpp>

using namespace std;

vector<string> load_list(const string& fname);

template<class T>
set<T> vector_to_set(const vector<T>& vec);

float compute_ap(const set<string>& pos, const set<string>& absent,
		const vector<string>& ranked_list, std::string prefix);

int main(int argc, char** argv) {

	if (argc != 3) {
		printf("\nUsage:\n\tComputeMAP "
				"<in.ranked.files.folder> <in.ranked.files.prefix>\n\n");
		return EXIT_FAILURE;
	}

	string ranked_files_folder = argv[1];
	string prefix = argv[2];

	float map = 0.0, ap;

	size_t i = 0;

	std::stringstream fname;
	while (true) {
		fname.str("");
		fname << ranked_files_folder << "/" << prefix << i;

		if (FileUtils::checkFileExist(fname.str() + "_ranked.txt") == false) {
			break;
		}

		printf("-- Computing average precision for query [%lu]\n", i);

		vector<string> ranked_list = load_list(fname.str() + "_ranked.txt");
		set<string> good_set = vector_to_set(
				load_list(fname.str() + "_good.txt"));
		set<string> ok_set = vector_to_set(load_list(fname.str() + "_ok.txt"));
		set<string> junk_set = vector_to_set(
				load_list(fname.str() + "_junk.txt"));

		set<string> pos_set;
		pos_set.insert(good_set.begin(), good_set.end());
		pos_set.insert(ok_set.begin(), ok_set.end());

		ap = compute_ap(pos_set, junk_set, ranked_list, fname.str());

#if CMAPVERBOSE
		printf("%f\n", ap);
#endif

		map += ap;

		++i;
	}

	printf("-- Mean average precision is [%f]\n", map / (float) i);

	return 0;
}

vector<string> load_list(const string& fname) {
	printf("   Loading list [%s]\n", fname.c_str());
	vector<string> ret;
	ifstream fobj(fname.c_str());
	if (!fobj.good()) {
		fprintf(stderr, "File [%s] not found!\n", fname.c_str());
		exit(-1);
	}
	string line;
	while (getline(fobj, line)) {
		ret.push_back(line);
	}
	return ret;
}

template<class T>
set<T> vector_to_set(const vector<T>& vec) {
	return set<T>(vec.begin(), vec.end());
}

float compute_ap(const set<string>& pos, const set<string>& absent,
		const vector<string>& ranked_list, std::string prefix) {

#if CMAPVERBOSE
	printf("size of positive set=%lu\n", pos.size());
#endif

	float old_recall = 0.0;
	float old_precision = 1.0;
	float ap = 0.0;

	prefix.append(".csv");
	FILE *f_ranked_list = fopen(prefix.c_str(), "w");
	std::stringstream precisionValues, recallValues;

	if (f_ranked_list == NULL) {
		fprintf(stderr, "Error opening file [%s] for writing\n",
				prefix.c_str());
		return EXIT_FAILURE;
	}

	size_t intersect_size = 0;
	size_t i = 0;
	size_t j = 0;
	for (; i < ranked_list.size(); ++i) {
		if (absent.count(ranked_list[i])) {
#if CMAPVERBOSE
			printf("%03lu) negative\n", i);
#endif
			// skip for the not relevant since they don't contribute
			// i.e. no area under the curve
			continue;
		} else if (pos.count(ranked_list[i])) {
#if CMAPVERBOSE
			printf("%03lu) relevant ", i);
#endif
			++intersect_size;
		} else {
#if CMAPVERBOSE
			printf("%03lu) notrelevant ", i);
#endif
		}

		// Divide over the number of relevant documents
		float recall = intersect_size / (float) pos.size();
		// Divide over the number of retrieved documents
		float precision = intersect_size / (j + 1.0);

		// instead of p(k) it does (p(k)+p(k-1))/2 times delta recall at k
		// i think to reduce the impact of wiggles in the curve
		float apAtK = (recall - old_recall)
				* ((old_precision + precision) / 2.0);

#if CMAPVERBOSE
		printf("precision=%f recall=%f \t ap=%f\n", precision, recall, apAtK);
#endif

		ap += apAtK;

		precisionValues << precision
				<< (((i + 1) < ranked_list.size()) ? "," : "");
		recallValues << recall << (((i + 1) < ranked_list.size()) ? "," : "");

		old_recall = recall;
		old_precision = precision;
		++j;
	}

	fprintf(f_ranked_list, "%s\n%s\n", precisionValues.str().c_str(),
			recallValues.str().c_str());
	fclose(f_ranked_list);

	return ap;
}
