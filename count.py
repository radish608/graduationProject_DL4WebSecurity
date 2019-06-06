#_*_coding:utf-8_*_

# 统计代码行数, 空行, 注释.

import os
def code_lines_count(path):
    code_lines = 0
    comm_lines = 0
    space_lines = 0
    for root,dirs,files in os.walk(path):
        for item in files:
            file_abs_path = os.path.join(root,item)
            postfix = os.path.splitext(file_abs_path)[1]
            if postfix == '.py':
                #print 'Start: ',file_abs_path
                with open(file_abs_path) as fp:
                    while True:
                        line = fp.readline()
                        if not line:
                            #print 'break here,%r' %line
                            break
                        elif line.strip().startswith('#'):
                            #print '1, here',line
                            comm_lines += 1
                        elif line.strip().startswith("'''") or line.strip().startswith('"""'):
                            comm_lines += 1
                            if line.count('"""') ==1 or line.count("'''") ==1:
                                while True:
                                    line = fp.readline()
                                    #print '4, here',line
                                    comm_lines += 1
                                    if ("'''" in line) or ('"""' in line):
                                        break
                        elif line.strip():
                            #print '5, here',line
                            code_lines += 1
                        else:
                            #print '6, here',line
                            space_lines +=1
                #print 'Done',file_abs_path
    return code_lines,comm_lines,space_lines
#test
print "Code lines: %d\nComments lines: %d\nWhiteSpace lines: %d" %code_lines_count(r'.')
