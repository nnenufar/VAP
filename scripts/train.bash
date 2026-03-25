python vap/main.py \
  datamodule.train_path=tests/data/splits/train_dset.csv \
  datamodule.val_path=tests/data/splits/val_dset.csv \
  datamodule.test_path=tests/data/splits/test_dset.csv \
  datamodule.batch_size=4 \
  datamodule.num_workers=4
