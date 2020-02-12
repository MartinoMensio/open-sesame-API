#!/bin/bash
set -e

virtualenv --python=python2 venv
source venv/bin/activate

# python requirements
pip install dynet
pip install nltk
python -m nltk.downloader averaged_perceptron_tagger wordnet

# requirement to download from Google Drive (requests is missing as dependency from here https://github.com/ndrplz/google-drive-downloader/)
pip install googledrivedownloader requests
# function to download with googledrivedownloader
gdrive_download() {
    python <<< "from google_drive_downloader import GoogleDriveDownloader as gdd;\
    gdd.download_file_from_google_drive(file_id='$1', dest_path='$2', showsize=True)"
}

# other requirements
mkdir -p data
pushd data
# https://drive.google.com/open?id=1s4SDt_yDhT8qFs1MZJbeFf-XeiNPNnx7
gdrive_download "1s4SDt_yDhT8qFs1MZJbeFf-XeiNPNnx7" "./fndata-1.7.tar.gz"
tar -xvzf fndata-1.7.tar.gz
rm fndata-1.7.tar.gz

wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
rm glove.6B.zip

# FrameNet 1.7 models
# https://drive.google.com/open?id=1sS0OPw1uYxeOUK0drkvfZsFkRNgnVUAC
gdrive_download "1sS0OPw1uYxeOUK0drkvfZsFkRNgnVUAC" "./fn1.7-pretrained-targetid.tar.gz"
# https://drive.google.com/open?id=1me1V0CrZF5HVWiDBqZ4LHZVSpsWfW3-8
gdrive_download "1me1V0CrZF5HVWiDBqZ4LHZVSpsWfW3-8" "./fn1.7-pretrained-frameid.tar.gz"
# https://drive.google.com/open?id=1ys-DIGhJSHgt8VjstMtlkPnYqtlzMSHe
gdrive_download "1ys-DIGhJSHgt8VjstMtlkPnYqtlzMSHe" "./fn1.7-pretrained-argid.tar.gz"


tar -xvzf fn1.7-pretrained-targetid.tar.gz
tar -xvzf fn1.7-pretrained-frameid.tar.gz
tar -xvzf fn1.7-pretrained-argid.tar.gz

# remove useless stuff
rm fn1.7-pretrained-targetid.tar.gz
rm fn1.7-pretrained-frameid.tar.gz
rm fn1.7-pretrained-argid.tar.gz

popd
python -m sesame.preprocess

# install the requirements for the API
pip install -r requirements.txt
