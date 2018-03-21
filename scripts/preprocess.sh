# preprocess the data

MOVIEDIR=$1
GDSTRTPT_LOC=$2

echo "[*] extracting frames from movies --> frames/";
python scripts/extract_frames.py $GDSTRTPT_LOC $MOVIEDIR frames/;

echo "[*] partitioning trials ---> partitions/, partitions.csv";
python scripts/partition_trials.py frames/ partitions/ partitions.csv;

echo "[*] resizing trials ---> rsz/";
python scripts/resize_trials.py partitions/ rsz/

echo "[*] consolidating partitions ---> consolidated/";
python scripts/consolidate_data.py rsz/

echo "[*] cataloguing partitions ---> catalogue.csv";
python scripts/catalog_partitions.py consolidated/ partitions.csv

echo "[*] creating labels csv ---> partitions_cons.csv";
python scripts/consolidated_to_csv.py consolidated
