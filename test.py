import json

def filter_source_pdf(input_file, output_file):
    """
    筛选JSON文件中 'source_pdf' 字段不为空的语料，并保存到新的JSON文件中。

    Args:
        input_file (str): 输入的JSON文件名。
        output_file (str): 输出的JSON文件名。
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：文件 '{input_file}' 未找到。")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 '{input_file}' 不是有效的JSON文件。")
        return

    filtered_data = [item for item in data if item.get('source_pdf')]

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)
        print(f"已成功筛选出 'source_pdf' 不为空的语料，并保存到 '{output_file}'。")
    except IOError:
        print(f"错误：无法写入文件 '{output_file}'。")

if __name__ == "__main__":
    input_filename = 'corpus/v1.0/data/Arts & Humanities_random_with_pdf_text.json'  # 将 'your_input_file.json' 替换为你的输入文件名
    output_filename = 'filtered_output.json'
    filter_source_pdf(input_filename, output_filename)