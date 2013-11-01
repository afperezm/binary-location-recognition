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
	cd VocabLearn; $(MAKE)
	cd VocabBuildDB; $(MAKE)
	cd VocabMatch; $(MAKE)
	cd ComputeMAP; $(MAKE)

clean:
# Clean libraries
	cd OpenCVExtensions; $(MAKE) clean
	cd Common; $(MAKE) clean
	cd KMajorityLib; $(MAKE) clean
	cd VocabLib; $(MAKE) clean
# Clean program
	cd FeatureExtract; $(MAKE) clean
	cd VocabLearn; $(MAKE) clean
	cd VocabBuildDB; $(MAKE) clean
	cd VocabMatch; $(MAKE) clean
	cd ComputeMAP; $(MAKE) clean
