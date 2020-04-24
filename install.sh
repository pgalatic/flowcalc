# Download Deepmatching and Deepflow
mkdir deepflow2
cd deepflow2
wget https://thoth.inrialpes.fr/src/deepmatching/code/deepmatching_1.2.2.zip
unzip deepmatching_1.2.2.zip
mv deepmatching_1.2.2_c++/deepmatching-static .
rm -r deepmatching_1.2.2_c++
rm deepmatching_1.2.2.zip
wget http://pascal.inrialpes.fr/data2/deepmatching/files/DeepFlow_release2.0.tar.gz
tar -xf DeepFlow_release2.0.tar.gz
mv DeepFlow_release2.0/deepflow2-static .
rm -r DeepFlow_release2.0
rm DeepFlow_release2.0.tar.gz
cd -

# Download and install SPyNet
git submodule update --init
cd spynet
git checkout master
pip3 install -r requirements.txt
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-spynet/network-chairs-clean.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-spynet/network-chairs-final.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-spynet/network-kitti-final.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-spynet/network-sintel-clean.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-spynet/network-sintel-final.pytorch
cd -

# Make ConsistencyChecker
cd consistencyChecker/
make
cd -