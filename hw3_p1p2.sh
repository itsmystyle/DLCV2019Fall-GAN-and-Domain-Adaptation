wget 'https://www.dropbox.com/s/oxld3vcdwxwpqyp/generator_537.pth.tar?dl=1' -O models/dcgan/generator_537.pth.tar
wget 'https://www.dropbox.com/s/zz6reezhuy4hu3a/generator_603.pth.tar?dl=1' -O models/acgan/generator_603.pth.tar
python3 generate_image.py --output_dir $1 --dcgan_model models/dcgan/generator_537.pth.tar --acgan_model models/acgan/generator_603.pth.tar --random_seed 42 821