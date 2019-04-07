# -*- coding: utf-8 -*-

import os
import traceback
import time
import codecs

def transferToUtf8(srcFilePath, dstFilePath):
    newlines = []
    max_line_length = 2048
    current_length = 0
    try:
        start = False
        for line in open(srcFilePath, 'r'):
            line = line.decode('GB2312').strip()
            current_length += len(line)
            if (current_length > max_line_length):
                return False
            if not start:
                start = isLineContainsChineseChar(line)
            if start and len(line) > 0:
                newlines.append(line.encode('utf-8') + " ")
    except UnicodeDecodeError, e:
        traceback.print_exc()
        return False
    if len(newlines) == 0:
        return False
    with open(dstFilePath, 'w') as f:
        for line in newlines:
            f.write(line)
    return True

def isLineContainsChineseChar(line):
    # return !((uchar >= '0' and uchar <= '9') or (uchar >= 'a' and uchar <= 'b') or (uchar >= 'a' and uchar <= 'b'))
    ret = False
    if len(line) > 0:
        ret = (line[0] >= u'\u4e00' and line[0]<=u'\u9fff')\
        or (line[-1] >= u'\u4e00' and line[-1]<=u'\u9fff')\
        or (line[len(line) / 2] >= u'\u4e00' and line[len(line) / 2]<=u'\u9fa5')
    return ret

def getIndexIter(indexFilePath):
    for line in open(indexFilePath, 'r'):
        lines = line.split(' ')
        lines[1] = lines[1][1:].strip()
        yield lines

def transerAllToUtf8(indexFilePath, outputDir):
    # Prepare dirs
    hamDir = outputDir + '/ham/'
    spamDir = outputDir + '/spam/'
    if not os.path.exists(outputDir):
        os.mkdir(outputDir)
    if not os.path.exists(hamDir):
        os.mkdir(hamDir)
    if not os.path.exists(spamDir):
        os.mkdir(spamDir)

    indexes = getIndexIter(indexFilePath)
    totalCount = 0
    errorCount = 0
    for index in indexes:
        if os.path.exists(index[1]):
            dstFilePath = spamDir if index[0] == 'spam' else hamDir
            dstFilePath += str(totalCount)
            print("{}:{}->{}".format(index[0], index[1], dstFilePath))
            if not transferToUtf8(index[1], dstFilePath):
                errorCount += 1
        else:
            print("{} not exists".format(index[1]))
        totalCount += 1
        # if totalCount == 100:
        #     break
    print("\ntotalCount = {}, errorCount = {}".format(totalCount, errorCount))

def getSingleLine(filepath):
    content = ""
    for line in open(filepath, 'r'):
        content = content + line
    # f = codecs.open(filepath,'r','utf-8')
    # lines = f.readlines()
    # for line in lines:
    # f.close()
    return content.strip() + '\n'

def concatenate_dir_to_file(dir_path, dst_filepath):
    with open(dst_filepath, 'w') as f:
        for parent, dirnames, filenames in os.walk(dir_path):
            print "parent is: " + parent
            for filename in filenames:
                full_filename = parent + '/' + filename
                f.write(getSingleLine(full_filename))

start_time = time.time()
# transerAllToUtf8("./full/index", './utf8data')
concatenate_dir_to_file("./utf8data/ham_5000", "./utf8data/ham_5000.utf8")
concatenate_dir_to_file("./utf8data/spam_5000", "./utf8data/spam_5000.utf8")
end_time = time.time()
print("used time : {} second".format(end_time - start_time))