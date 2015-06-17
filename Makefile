# Top-level Makefile

DEBUGFLAGS = -g -Wall
#export DEBUGFLAGS

all: libs programs

libs:
# Make libraries
	cd OpenCVExtensions; $(MAKE)
	cd Common; $(MAKE)
	cd KMajorityLib; $(MAKE)
	cd IncrementalKMeansLib; $(MAKE)
	cd VocabLib; $(MAKE)

programs:
# Make programs
	cd FeatureExtract; $(MAKE)
	cd FeatureSelect; $(MAKE)
	cd SelectDescriptors; $(MAKE)
	cd VocabLearn; $(MAKE)
	cd VocabBuildDB; $(MAKE)
	cd VocabMatch; $(MAKE)
	cd GeomVerify; $(MAKE)
	cd ComputeMAP; $(MAKE)
	cd ListBuild; $(MAKE)

clean: clean-libs clean-programs

clean-libs:
# Clean libraries
	cd OpenCVExtensions; $(MAKE) clean
	cd Common; $(MAKE) clean
	cd KMajorityLib; $(MAKE) clean
	cd IncrementalKMeansLib; clean
	cd VocabLib; $(MAKE) clean

clean-programs:
# Clean program
	cd FeatureExtract; $(MAKE) clean
	cd FeatureSelect; $(MAKE) clean
	cd SelectDescriptors; $(MAKE) clean
	cd VocabLearn; $(MAKE) clean
	cd VocabBuildDB; $(MAKE) clean
	cd VocabMatch; $(MAKE) clean
	cd GeomVerify; $(MAKE) clean
	cd ComputeMAP; $(MAKE) clean
	cd ListBuild; $(MAKE) clean

tests:
#	cd Common; $(MAKE)
#	cd KMajorityLib; $(MAKE)
#	cd VocabLib; $(MAKE)
	cd Common/tests; $(MAKE)
	cd KMajorityLib/tests; $(MAKE)
	cd IncrementalKMeansLib/tests; $(MAKE)
	cd VocabLib/tests; $(MAKE)

tests-clean:
#	cd Common; $(MAKE) clean
#	cd KMajorityLib; $(MAKE)
#	cd VocabLib; $(MAKE) clean
	cd Common/tests; $(MAKE) clean
	cd KMajorityLib/tests; $(MAKE) clean
	cd IncrementalKMeansLib/tests; clean
	cd VocabLib/tests; $(MAKE) clean
