import os
import pandas as pd

root = "Aishell_dataset/"

txt_label_path = "data_aishell/transcript/aishell_transcript_v0.8.txt"


def traverse(root, path, search_fix=".txt"):
    f_list = []

    p = root + path
    for s_p in sorted(os.listdir(p)):
        for sub_p in sorted(os.listdir(p + "/" + s_p)):
            if sub_p[len(sub_p) - len(search_fix):] == search_fix:
                print(">", path, s_p, sub_p)
                f_list.append(p + "/" + s_p + "/" + sub_p)

    return f_list


with open(txt_label_path, "r",encoding="utf-8") as f:
    lines = f.readlines()

# alllabel = []
filenames = []
labels = []
for line in lines:
    j = 0
    filename = line.strip().split(' ')[0]
    label = line.strip().split(' ')[1:-1]
    newlabel = ""
    flag = 0
    for i in label:
        if i != '':
            flag = 1
        if flag == 1:
            newlabel = newlabel + i + " "
    # print(newlabel)
    # alllabel.append([filename,newlabel])
    filenames.append(filename)
    labels.append(newlabel)

alllabel = pd.DataFrame({'filename': filenames, 'label': labels}, index=None)

print(alllabel.head())
print(str(alllabel[alllabel["filename"] == 'BAC009S0002W0122']["label"].values))
tr_file_list = traverse(root, "transcript/train", search_fix=".wav")
dev_file_list = traverse(root, "transcript/dev", search_fix=".wav")
test_file_list = traverse(root, "transcript/test", search_fix=".wav")
# 划分标签
for i in range(len(tr_file_list)):
    text_file_path = tr_file_list[i]
    fname = text_file_path.split("/")[-1].replace(".wav", "")
    new_text_file_path = tr_file_list[i].replace(".wav", ".docx")
    flabel = str(alllabel[alllabel["filename"] == fname]["label"].values).replace("['", "").replace("']", "")
    print(new_text_file_path, flabel)
    with open(new_text_file_path, "w", encoding="utf-8") as new_text_file:
        new_text_file.write(flabel + "\n")


for i in range(len(dev_file_list)):
    text_file_path = dev_file_list[i]
    fname = text_file_path.split("/")[-1].replace(".wav", "")
    new_text_file_path = dev_file_list[i].replace(".wav", ".docx")
    flabel = str(alllabel[alllabel["filename"] == fname]["label"].values).replace("['", "").replace("']", "")
    print(new_text_file_path, flabel)
    with open(new_text_file_path, "w", encoding="utf-8") as new_text_file:
        new_text_file.write(flabel + "\n")


for i in range(len(test_file_list)):
    text_file_path = test_file_list[i]
    fname = text_file_path.split("/")[-1].replace(".wav", "")
    new_text_file_path = test_file_list[i].replace(".wav", ".docx")
    flabel = str(alllabel[alllabel["filename"] == fname]["label"].values).replace("['", "").replace("']", "")
    print(new_text_file_path, flabel)
    with open(new_text_file_path, "w", encoding="utf-8") as new_text_file:
        new_text_file.write(flabel + "\n")

