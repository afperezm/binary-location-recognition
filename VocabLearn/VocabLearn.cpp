/*
 * VocabLearn.cpp
 *
 *  Created on: Sep 30, 2013
 *      Author: andresf
 */

#include <stdlib.h>
#include <VocabTree.h>

int main(int argc, char **argv) {

	if (argc != 6) {
		printf("Usage: %s <list.in> <depth> <branching_factor> "
				"<restarts> <tree.out>\n", argv[0]);
		return EXIT_FAILURE;
	}

	const char *list_in = argv[1];
	int depth = atoi(argv[2]);
	int branchFactor = atoi(argv[3]);
	int restarts = atoi(argv[4]);
	const char *tree_out = argv[5];
	printf("Building tree with depth: %d, branching factor: %d, "
			"and restarts: %d\n", depth, branchFactor, restarts);

	// Read key files

	// Compute total number of keys
//	printf("Total number of keys: %lu\n", total_keys);
	fflush(stdout);

	cvflann::VocabTree tree;
	tree.build();
//	Build(total_keys, dim, depth, bf, restarts, vp);
//	tree.Write(tree_out);

	return EXIT_SUCCESS;
}
