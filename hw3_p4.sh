wget 'https://www.dropbox.com/s/p11qsy14bii6qm6/dsn_m2s_model_best.pth.tar?dl=1' -O models/dsn_m2s/model_best.pth.tar
wget 'https://www.dropbox.com/s/bckqxvktu470bed/dsn_s2m_model_best.pth.tar?dl=0' -O models/dsn_s2m/model_best.pth.tar
python3 module/dsn/predict.py $1 $2 $3