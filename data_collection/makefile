
# Number of times to run the script
NUM_RUNS=100

# Target to run the script multiple times
run:
	@for i in $(shell seq 1 $(NUM_RUNS)); do \
		echo "Run #$$i"; \
		python data_collection.py; \
	done
