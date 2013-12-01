/*
 * DirectIndex.h
 *
 *  Created on: Nov 30, 2013
 *      Author: andresf
 */

#ifndef DIRECTINDEX_H_
#define DIRECTINDEX_H_

#include <cstring>
#include <map>
#include <vector>

namespace bfeat {

typedef std::vector<int> FeatureVector;
typedef std::map<int, FeatureVector> TreeNode;
typedef unsigned int uint;

class DirectIndex {

protected:
	// Level at which nodes are stored to construct the direct index
	int m_level;

	// Vector holding the index of images
	std::vector<TreeNode> m_index;

public:

	/**
	 * Class constructor.
	 *
	 * @param level - level of the tree at which nodes are stored
	 */
	DirectIndex(int level);

	/**
	 * Class destroyer
	 */
	virtual ~DirectIndex();

	/**
	 * Returns the number of images in the index.
	 *
	 * @return the index size
	 */
	size_t size() const;

	/**
	 * Return the level of the tree at which nodes are stored
	 *
	 * @return level
	 */
	int getLevel() const;

	/**
	 * Updates the direct index of the given image by
	 * pushing the feature to its corresponding node.
	 *
	 * @param imgIdx
	 * @param nodeId
	 * @param featureId
	 */
	void addFeature(uint imgIdx, int nodeId, int featureId);

};

} /* namespace bfeat */

#endif /* DIRECTINDEX_H_ */
