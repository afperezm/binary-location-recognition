# Top-level Makefile

all: default

default:
# Make libraries
	cd OpenCVExtensions; $(MAKE)
	cd Common; $(MAKE)
	cd KMajorityLib; $(MAKE)
	cd VocabLib; $(MAKE)
# Make program
	cd FeatureExtract; $(MAKE)
	cd FeatureSelect; $(MAKE)
	cd SelectDescriptors; $(MAKE)
	cd VocabLearn; $(MAKE)
	cd VocabBuildDB; $(MAKE)
	cd VocabMatch; $(MAKE)
	cd GeomVerify; $(MAKE)
	cd ComputeMAP; $(MAKE)

clean:
# Clean libraries
	cd OpenCVExtensions; $(MAKE) clean
	cd Common; $(MAKE) clean
	cd KMajorityLib; $(MAKE) clean
	cd VocabLib; $(MAKE) clean
# Clean program
	cd FeatureExtract; $(MAKE) clean
	cd FeatureSelect; $(MAKE) clean
	cd SelectDescriptors; $(MAKE) clean
	cd VocabLearn; $(MAKE) clean
	cd VocabBuildDB; $(MAKE) clean
	cd VocabMatch; $(MAKE) clean
	cd GeomVerify; $(MAKE) clean
	cd ComputeMAP; $(MAKE) clean

tests:
#	cd Common; $(MAKE)
#	cd KMajorityLib; $(MAKE)
#	cd VocabLib; $(MAKE)
	cd Common/tests; $(MAKE)
	cd KMajorityLib/tests; $(MAKE)
	cd VocabLib/tests; $(MAKE)

tests-clean:
#	cd Common; $(MAKE) clean
#	cd KMajorityLib; $(MAKE)
#	cd VocabLib; $(MAKE) clean
	cd Common/tests; $(MAKE) clean
	cd KMajorityLib/tests; $(MAKE) clean
	cd VocabLib/tests; $(MAKE) clean
