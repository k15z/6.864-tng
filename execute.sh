python train_cnn.py --hidden_size 256 --pooling mean > logs/cnn.256.mean.log
python train_cnn.py --hidden_size 256 --pooling max > logs/cnn.256.max.log
python train_cnn.py --hidden_size 256 --pooling alpha > logs/cnn.256.alpha.log
python train_cnn_da.py --hidden_size 256 --pooling mean > logs/cnn_da.256.mean.log

python train_cnn.py --hidden_size 512 --pooling mean > logs/cnn.512.mean.log
python train_cnn.py --hidden_size 512 --pooling max > logs/cnn.512.max.log
python train_cnn.py --hidden_size 512 --pooling alpha > logs/cnn.512.alpha.log
python train_cnn_da.py --hidden_size 512 --pooling mean > logs/cnn_da.512.mean.log
