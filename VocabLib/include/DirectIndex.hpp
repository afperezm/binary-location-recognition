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
#include <stdexcept>
#include <vector>

#include <opencv2/core/core.hpp>

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
	DirectIndex(int level = -1);

	/**
	 * Class destroyer.
	 */
	virtual ~DirectIndex();

	/**
	 * Returns the number of images in the index.
	 *
	 * @return the index size
	 */
	size_t size() const;

	/**
	 * Return the level of the tree at which nodes are stored.
	 *
	 * @return level
	 */
	int getLevel() const;

	void setLevel(int level);

	/**
	 * Updates the direct index of the given image by
	 * pushing the feature to its corresponding node.
	 *
	 * @param imgIdx
	 * @param nodeId
	 * @param featureId
	 */
	void addFeature(int imgIdx, int nodeId, int featureId);

	const TreeNode& lookUpImg(int imgIdx) const;

	void save(const std::string& filename) const;

	void load(const std::string& filename);

	void clear() const;

};

} /* namespace bfeat */

#endif /* DIRECTINDEX_H_ */
