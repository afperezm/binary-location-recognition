/*
 * VocabBuildDB.cpp
 *
 *  Created on: Oct 9, 2013
 *      Author: andresf
 */

#include <stdlib.h>

int main(int argc, char **argv) {

    if (argc < 4 || argc > 8) {
        printf("Usage: %s <in.list> <in.tree> <out.tree> [use_tfidf:1] "
               "[normalize:1] [start_id:0] [distance_type:1]\n",
               argv[0]);

        return 1;
    }

}

