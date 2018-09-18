import os

root_dir = "/mnt/lustre/niuyazhe/nyz/TP-GAN_data/"
pics_list = ["15","30","45","60","75"]
file_list = "tp_gan_all.txt"
f = open(file_list, "w")


for ele in pics_list:
    path = os.path.join(root_dir,ele)
    for e in os.listdir(path):
        input_data = os.path.join(path, e)
        if e[-8:-4] == "test":  # is input
            gt = os.path.join(path,e[:-9]+".png")
            if os.path.exists(gt):
                f.write(input_data+"\t"+gt+"\n")
f.close()
