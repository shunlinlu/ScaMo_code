
mkdir -p pretrained
cd pretrained/

echo -e "The pretrained model files will be stored in the 'pretrained' folder\n"
gdown 1EczHrsU7LsmCZ5KTPFAE-31OvYAUZrVq

unzip ScaMo_3B.zip
echo -e "Cleaning\n"
rm ScaMo_3B.zip

echo -e "Downloading done!"