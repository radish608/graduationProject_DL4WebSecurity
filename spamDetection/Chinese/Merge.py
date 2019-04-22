# encoding: utf-8

import os

for dir_name in os.listdir('./'):
    if os.path.isdir(dir_name):
        f_out = open(dir_name+'.txt','w')
        for file_name in os.listdir(dir_name):
            f_out.write(open(os.path.join(dir_name,file_name),'r').read()+'\n')
        f_out.close()
