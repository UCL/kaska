#!/bin/bash
# To add changes in this file to a running VM, use command:
# vagrant reload --provision 
echo "*** Provisioning... ***"

# Install dh and other useful tools:
apt-get -y update
#cat <<EOF > /etc/apt/apt.conf.d/local

#Dpkg::Options {
#   "--force-confdef";
#   "--force-confold";
#};

#EOF

#DEBIAN_FRONTEND=noninteractive apt-get -yq install openssh-server
#apt-get -y upgrade
#apt-get -y install devscripts debmake git cmake

# Set locale using method from:
# https://www.thomas-krenn.com/en/wiki/Perl_warning_Setting_locale_failed_in_Debian
locale-gen en_GB.UTF-8
localedef -i en_GB -f UTF-8 en_GB.UTF-8
# NB another method (I don't recall where from) works if you do:
# echo "export LC_ALL=C" >> .bashrc ; source ./.bashrc

# Kaska set up
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
sha256sum ./Anaconda3-2019.10-Linux-x86_64.sh

# Run these manually (licence agreemewnt prompt issue)
# bash ./Anaconda3-2019.10-Linux-x86_64.sh
# source ~/.bashrc
# source anaconda3/etc/profile.d/conda.sh
# export PATH=${PATH}:anaconda3/bin
# git clone https://github.com/UCL/kaska.git
# cd kaska 
# conda create -n kaska python=3.7 tensorflow numba gdal
# conda activate kaska
# pip install -e .

# Something to make my life easier during development:
echo "alias h=history" >> ~vagrant/.bashrc

echo "*** Provisioning completed. ***"

