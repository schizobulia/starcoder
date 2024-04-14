from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "bigcode/starcoder2-7b"
device = "cuda" # for GPU usage or "cpu" for CPU usage

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# for multiple GPUs install accelerate and do `model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto")`
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt").to(device)


outputs = model.generate(inputs)
print('=======1=======')
print(tokenizer.decode(outputs[0]))
print('=======2=======')

# import os
# from pathlib import Path
# def read_files_in_directory(directory, file_extension, new_directory, new_extension):
#     # 遍历目录和子目录
#     for root, dirs, files in os.walk(directory):
#         # 遍历当前目录下的所有文件
#         for file in files:
#             # 构建完整的文件路径
#             full_file_path = os.path.join(root, file)
#             # 检查文件是否为目录，并且文件名是否以指定的后缀名结尾
#             if not os.path.isdir(full_file_path) and file.endswith(file_extension):
#                 # 打开文件并打印内容
#                 with open(full_file_path, 'r', encoding='utf-8') as f:
#                     relative_path = os.path.relpath(full_file_path, directory)
#                     new_file_path = os.path.join(new_directory, relative_path)
#                     f_p_directory = os.path.dirname(new_file_path)
#                     file_name = Path(new_file_path).stem
#                     writeNewfile(os.path.join(f_p_directory, f"{file_name}.{new_extension}"), f.read())
#                     print(file_name)
#                     # print(f.read())
#                     # print("-" * 40)  # 打印分隔线

# def writeNewfile(file_path, content):
#     # 获取文件所在的上层目录
#     directory = os.path.dirname(file_path)
    
#     # 创建上层目录（如果它不存在）
#     if not os.path.exists(directory):
#         os.makedirs(directory)
    
#     # 检查文件是否存在
#     if os.path.exists(file_path) and os.path.isfile(file_path):
#         print('file is exist')
#     else:
#         # 文件不存在，创建并写入内容
#         with open(file_path, 'w', encoding='utf-8') as f:
#             f.write(content)


# # 指定要遍历的目录路径和文件后缀名
# directory_path = "/Users/gongyanan/tmp/vdom/src"  # 请替换为你的目录路径
# file_extension = ".js"  # 指定文件后缀名
# new_directory_path = "/Users/gongyanan/tmp/vdom/www"
# new_extension = ".java"

# # 调用函数并打印结果
# read_files_in_directory(directory_path, file_extension, new_directory_path, new_extension)


