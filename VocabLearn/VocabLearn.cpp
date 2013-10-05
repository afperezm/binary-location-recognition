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

	cvflann::VocabTree tree;
	tree.build();
//	Build(total_keys, dim, depth, bf, restarts, vp);
//	tree.Write(tree_out);

	return EXIT_SUCCESS;
}
