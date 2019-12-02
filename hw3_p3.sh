wget 'https://www.dropbox.com/s/3cog4s55kp7rlty/dann_m2s_model_best.pth.tar?dl=1' -O models/dann_m2s/model_best.pth.tar
wget 'https://www.dropbox.com/s/argeu74sx66immu/dann_s2m_model_best.pth.tar?dl=1' -O models/dann_s2m/model_best.pth.tar
python3 predict_dann.py $1 $2 $3