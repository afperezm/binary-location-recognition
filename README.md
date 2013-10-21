# BinVocabTree #

## Overview ##

Implementation of my solution to the problem of image-based location recognition using binary descriptors.

## License ##

This code is distributed under a MIT license. See the source code for details.

## Components ##

__Common:__ holds utility functions and classes for file and folder reading/write among others.

__OpenCVExtensions:__ wrapper for detectors/descriptors authors implementations.

__FeatureExtract:__ program for detecting and describing keypoints from a set of JPG images located in a folder. 

__KMajorityLib:__ implementation of k-majority clustering algorithm.

__VocabLib:__ implementation of vocabulary tree algorithm.

__VocabLearn:__ program for constructing a vocabulary tree by running hierarchical clustering over data using k-majority at each level.

__VocabBuildDB:__ program for loading inverted files .

__VocabMatch:__

## How to run ##

```
Usage:
		FeatureExtract <in.imgs.folder> <out.keys.folder>
Arguments:
		<in.imgs.folder> folder where to find JPG images to extract features from.
		<out.keys.folder> output folder where to save the extracted features. 
```

```
Usage:
		VocabLearn <in.list> <in.depth> <in.branch.factor> <in.restarts> <out.tree>
Arguments:
		<in.list> plain text file with a newline separated list of images.
		<in.depth> maximum number of levels in the tree.
		<in.branch.factor> number of clusters each node should be splitted.
		<in.restarts> maximum number of clustering iterations.
		<out.tree> file with .yaml.gz or .xml.gz extension where the vocabulary tree will be saved.
```

```
Usage:
		VocabBuildDB <in.list> <in.tree> <out.tree> [in.use.tfidf:1] [in.normalize:1]
Arguments:
		<in.list> plain text file with a line separated list of DB images.
		<in.tree> file with .yaml.gz or .xml.gz extension containing the vocabulary tree structure with empty inverted files.
		<out.tree> file with .yaml.gz or .xml.gz extension containing the vocabulary tree taken as input with loaded inverted files and words weights. 
		[in.use.tfidf:1] flag indicating word weighting scheme to be used, true by default for TF-IDF, false for a constant weight.
		[in.normalize:1] flag indicating if DB BoW vectors should be normalized, true by default.
```

```
Usage:
		VocabMatch <in.tree> <in.db.gt.list> <in.query.list> <in.num.neighbors> <out.matches> [out.visual.results:results.html] [out.candidates:candidates.txt]
Arguments:
		<in.tree> file with .yaml.gz or .xml.gz extension containing the vocabulary tree structure with loaded inverted files and words weights.
		<in.db.gt.list> plain text file with a newline separated list of DB images with landmark id formatted as <key.file> <landmark.id>.
		<in.query.list> plain text file with a newline separated list of query images.
		<in.num.neighbors> top number of results to retrieve.
		<out.matches> plain text file where the top voted results for each query image are stored in the format <query.id> <max.voted.landmark.id> <number.votes>.
		[out.results:results.html] HTML formatted file with the retrieval results showing images and scores.
		[out.candidates:candidates.txt] plain text file with a newline separated list of retrieval results (images full path) ordered by score.
```

## C++ HTTP library options ##

Starred ones:

[cppserv](http://www.total-knowledge.com/progs/cppserv/) 

[acl_cpp](sourceforge.net/projects/aclcpp/)

[libmicrohttp](http://www.gnu.org/software/libmicrohttpd/)

Others:

[cpp-netlib](https://github.com/cpp-netlib/cpp-netlib)

[mongoose](https://code.google.com/p/mongoose/](https://code.google.com/p/mongoose/)

[http-parser](https://github.com/joyent/http-parser)
