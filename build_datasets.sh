python download.py
python split_dataset.py --dataset_dir data/annotations
python map_classes.py --annotation_path data/annotations/annotations_0_train.json --mapping_path data/mappings/map_10.csv
python map_classes.py --annotation_path data/annotations/annotations_0_val.json --mapping_path data/mappings/map_10.csv
python map_classes.py --annotation_path data/annotations/annotations_0_test.json --mapping_path data/mappings/map_10.csv