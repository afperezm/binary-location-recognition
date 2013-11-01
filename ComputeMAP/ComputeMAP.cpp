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
#include <cstdlib>

#include <FileUtils.hpp>

using namespace std;

vector<string> load_list(const string& fname);

template<class T>
set<T> vector_to_set(const vector<T>& vec);

float compute_ap(const set<string>& pos, const set<string>& absent,
		const vector<string>& ranked_list);

int main(int argc, char** argv) {

	if (argc != 3) {
		printf("\nUsage:\n"
				"\tComputeMAP <ground.truth.folder> <queries.prefix>\n\n");
		return EXIT_FAILURE;
	}

	string gt_folder = argv[1];
	string qprefix = argv[2];

	float map = 0.0, ap;

	size_t i = 0;

	std::stringstream fname;
	while (true) {
		fname.str("");
		fname << gt_folder << "/" << qprefix << i;

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

		ap = compute_ap(pos_set, junk_set, ranked_list);

		map += ap;

		i++;
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
		const vector<string>& ranked_list) {
	float old_recall = 0.0;
	float old_precision = 1.0;
	float ap = 0.0;

	size_t intersect_size = 0;
	size_t i = 0;
	size_t j = 0;
	for (; i < ranked_list.size(); ++i) {
		if (absent.count(ranked_list[i])) {
			continue;
		}
		if (pos.count(ranked_list[i])) {
			intersect_size++;
		}

		float recall = intersect_size / (float) pos.size();
		float precision = intersect_size / (j + 1.0);

		ap += (recall - old_recall) * ((old_precision + precision) / 2.0);

		old_recall = recall;
		old_precision = precision;
		j++;
	}
	return ap;
}
