python train_cnn.py --hidden_dims 256 --pooling mean > logs/cnn.256.mean.log
python train_cnn.py --hidden_dims 256 --pooling max > logs/cnn.256.max.log

python train_cnn_da.py --hidden_dims 256 --pooling mean > logs/cnn_da.256.mean.log
