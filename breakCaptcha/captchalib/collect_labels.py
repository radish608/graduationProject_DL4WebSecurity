# -*- coding:utf-8 -*-
import os
import json

root_dir = "/home/randolph1997/DL4WebSecurity/breakCaptcha/captchalib/dataset"
file_list = os.listdir(root_dir)

label_set = set()

for file in file_list:
    split_result = file.split("_")
    if len(split_result) == 2:
        label_list, name = split_result
        if label_list:
            for label in label_list:
                label_set.add(label)
    else:
        pass

print "共有{}个标签".format(len(label_set))

with open("/home/randolph1997/DL4WebSecurity/breakCaptcha/captchalib/labels.json", "w") as jf:
    jf.write(json.dumps(list(label_set))) #如果有中文，要改成json.dumps(label Set, ensure_ascii=False)
