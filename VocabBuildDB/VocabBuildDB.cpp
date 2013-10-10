/*
 * VocabBuildDB.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <stdlib.h>

int main(int argc, char **argv) {

	if (argc < 4 || argc > 7) {
		printf("Usage: %s <in.list> <in.tree> <out.tree> [use_tfidf:1] "
				"[normalize:1] [start_id:0] [distance_type:1]\n", argv[0]);
		return EXIT_FAILURE;
	}

	bool use_tfidf = true;
	bool normalize = true;

	char *list_in = argv[1];
	char *tree_in = argv[2];
	char *tree_out = argv[3];

	if (argc >= 5) {
		use_tfidf = atoi(argv[4]);
	}

	if (argc >= 6) {
		normalize = atoi(argv[5]);
	}

//    if (argc >= 7)
//	distance_type = (DistanceType) atoi(argv[7]);
//
//	switch (distance_type) {
//	case DistanceDot:
//		printf("[VocabMatch] Using distance Dot\n");
//		break;
//	case DistanceMin:
//		printf("[VocabMatch] Using distance Min\n");
//		break;
//	default:
//		printf("[VocabMatch] Using no known distance!\n");
//		break;
//	}

	// Read keys

	//

	printf("-- \n");
}

