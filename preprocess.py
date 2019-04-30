# coding:utf-8

# @Created : Macielyoung
# @Time : 2019/4/18
# @Function : preprocess the raw text

Data_File = "data/conversations.txt"
sos, eos, pad = "SOS", "EOS", "PAD"

def read_file(file):
    return open(file, 'r').readlines()

def max_line(lines):
    max_len = 0
    for line in lines:
        length = len(line.split(" "))
        if length > max_len:
            max_len = length
    return max_len

lines = read_file(Data_File)
maxLen = max_line(lines)
# print(max_len)

def padding_sentence(lines, maxLen):
    new_lines = []
    for line in lines:
        line = line.strip()
        length = len(line.split(" "))
        new_list = [sos] + line.split(" ") + (maxLen-length) * [pad] + [eos]
        new_line = " ".join(new_list)
        new_lines.append(new_line)
    return new_lines

def write_txt(new_lines):
    lengths = [len(line.split(" ")) for line in new_lines]
    lines = [line+"\n" for line in new_lines]
    with open("padding.txt", "w") as f:
        f.writelines(lines)
    return lengths

new_lines = padding_sentence(lines, maxLen)
print(new_lines)

lengths = write_txt(new_lines)
print(lengths)
ser_len = set(lengths)
print(ser_len)
