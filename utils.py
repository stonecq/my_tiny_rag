import json


def read_json_to_list(file_path):
    """
    从JSON文件读取数据到列表中。

    参数:
    file_path (str): JSON文件的路径

    返回:
    list: 包含JSON文件中数据的列表
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl_to_list(file_path):
    """
    从 JSONL (JSON Lines) 文件读取数据到列表中。
    每一行必须是一个有效的 JSON 对象。

    参数:
    file_path (str): JSONL 文件的路径

    返回:
    list: 包含所有解析后的字典对象的列表
    """
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            json_obj = json.loads(line)
            data_list.append(json_obj)
    return data_list

def write_list_to_jsonl(data_list, path):
    with open(path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")