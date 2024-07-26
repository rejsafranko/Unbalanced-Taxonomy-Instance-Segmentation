python src/data/download.py
python src/data/split_dataset.py --dataset_dir data/annotations
python src/data/map_classes.py --annotation_path data/annotations/annotations_0_train.json --mapping_path data/mappings/map_10.csv
python src/data/map_classes.py --annotation_path data/annotations/annotations_0_val.json --mapping_path data/mappings/map_10.csv
python src/data/map_classes.py --annotation_path data/annotations/annotations_0_test.json --mapping_path data/mappings/map_10.csv