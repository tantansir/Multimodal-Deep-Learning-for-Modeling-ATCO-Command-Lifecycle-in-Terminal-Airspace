import pandas as pd
import re

# process Voice.csv
# Load the uploaded Excel file
file_path = 'data/sample/Voice_25-11-2022_refined_timestamp.xlsx'
df = pd.read_excel(file_path)

# Add a new column 'cancel' with default value 0
df['cancel'] = ''

# 地点名称
place_names = ["Ignon", "lavax", "sanat", "nylon", "paspu", "dovan", "bipop", "posub", "ruvik", "ibula"] #rubik*是什么鬼

# 定义动作模式
action_patterns = [
    r"cancel\s+((?:\w+\s+)?(?:restriction|hold|altitude restriction|height|level restriction))",  # 例：cancel altitude restriction
    r"(\w+\s+(?:restriction|hold|altitude restriction|height|level restriction))\s+cancel",  # 例：altitude restriction cancel
    r"(\w+\s+(?:restriction|hold|altitude restriction|height|level restriction))\s+cancelled"  # 例：height cancelled
]

# 提取动作的函数，包括地点名
def extract_action_with_place(line):
    for place in place_names:
        # 查找"地点 + 动作"的模式
        place_pattern = fr"{place}\s+(\w+\s+(?:restriction|hold|altitude restriction|height|level restriction))"
        match = re.search(place_pattern, line, re.IGNORECASE)
        if match:
            return f"{place} {match.group(1)}"
    # 如果没有匹配地点名，则使用原有动作提取逻辑
    for pattern in action_patterns:
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# 找出Source列包含'C'且Lines列包含'cancel'字样的行
cancel_condition = df['Source'].str.contains('C') & df['Lines'].str.contains('cancel', case=False)

# 应用动作提取逻辑
df.loc[cancel_condition, 'cancel'] = df[cancel_condition]['Lines'].apply(extract_action_with_place)

# 找出Source列包含'C'且Lines列包含'maintain'字样，且condition列不包含'after'的行
maintain_condition = (
    df['Source'].str.contains('C') &
    df['Lines'].str.contains('maintain', case=False) &
    ~df['condition'].str.contains('after', case=False, na=False)
)

# 添加 'maintain' 列，并根据条件设置值
df['maintain'] = ''
df.loc[maintain_condition, 'maintain'] = 1

# 删除 'Matched' 列
df.drop(columns=['Matched'], inplace=True)

# 保存为CSV文件
output_file_path = 'Voice.csv'
df.to_csv(output_file_path, index=False)