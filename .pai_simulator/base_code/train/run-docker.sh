docker build -t pai-simulator-base_code-train '/home/nada/masterarbeit/VesselSeg-Pytorch-Semi-supervised/.pai_simulator/base_code/train'
docker run --rm pai-simulator-base_code-train
docker rmi pai-simulator-base_code-train
read -p "Press [Enter] to continue ..."