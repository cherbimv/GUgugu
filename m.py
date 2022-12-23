import glob

file_list = glob.glob("./source/*")
print(file_list)

with open('data/result.txt', 'w', encoding='utf-8') as file:
    for i in file_list:
        with open(i, 'r', encoding='utf-8') as f:
            input_lines = f.readlines()
            file.writelines(input_lines)