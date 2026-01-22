import json
import re
import pandas as pd
from datetime import datetime
import os

def extract_actions_from_predict(predict_str):
    """从predict字符串中提取动作信息"""
    actions = []
    
    # 检查是否有多个动作（用" and "分隔）
    if " and " in predict_str:
        action_parts = predict_str.split(" and ")
    else:
        action_parts = [predict_str]
    
    for action in action_parts:
        # 提取标签
        label_match = re.search(r'<driving_maneuver>(.*?)</driving_maneuver>', action)
        # 提取开始时间
        start_match = re.search(r'<start_time>(.*?)</start_time>', action)
        # 提取结束时间
        end_match = re.search(r'<end_time>(.*?)</end_time>', action)
        
        if label_match and start_match and end_match:
            actions.append({
                'label': label_match.group(1),
                'start_time': float(start_match.group(1)),
                'end_time': float(end_match.group(1))
            })
    
    return actions

def generate_validation_excel_with_video_paths(predict_jsonl_path, dataset_json_path, output_excel_path=None):
    """
    生成带视频路径的验证Excel文件
    
    Args:
        predict_jsonl_path: 预测结果JSONL文件路径
        dataset_json_path: 原始数据集JSON文件路径
        output_excel_path: 输出Excel文件路径（可选）
    """
    
    if output_excel_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_excel_path = f'validation_with_videos_{timestamp}.xlsx'
    
    # 1. 读取原始数据集，提取视频路径
    print(f"正在读取原始数据集: {dataset_json_path}")
    with open(dataset_json_path, 'r', encoding='utf-8') as f:
        dataset_data = json.load(f)
    
    # 提取视频路径列表
    video_paths = []
    for item in dataset_data:
        if 'videos' in item and len(item['videos']) > 0:
            video_paths.append(item['videos'][0])  # 取第一个视频
        else:
            video_paths.append('N/A')
    
    print(f"原始数据集共 {len(video_paths)} 个样本")
    
    # 2. 读取预测结果，并与视频路径对应
    all_actions = []
    line_num = 0
    
    with open(predict_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 0):  # 从0开始计数，对应视频路径的索引
            try:
                data = json.loads(line.strip())
                predict_str = data.get('predict', '')
                true_label = data.get('label', '').strip()
                
                # 获取对应的视频路径
                if line_num < len(video_paths):
                    video_path = video_paths[line_num]
                else:
                    video_path = f"超出范围_行{line_num}"
                    print(f"警告: 预测结果第{line_num}行没有对应的视频路径")
                
                # 提取动作
                actions = extract_actions_from_predict(predict_str)
                
                if not actions:
                    # 如果没有提取到动作，也记录一行
                    all_actions.append({
                        'BOS路径': video_path,
                        '预测标签': '无预测结果',
                        '真实标签': true_label,
                        '是否一致': '',
                        '开始时间': '',
                        '结束时间': '',
                        '备注': '未提取到有效预测结果',
                        '原predict': predict_str,
                        'jsonl行号': line_num + 1  # 显示为1-based
                    })
                else:
                    for action in actions:
                        all_actions.append({
                            'BOS路径': video_path,
                            '预测标签': action['label'],
                            '真实标签': true_label,
                            '是否一致': '',  # 留空，用于人工填写
                            '开始时间': action['start_time'],
                            '结束时间': action['end_time'],
                            '备注': '',  # 用于填写不一致的原因
                            '原predict': predict_str,
                            'jsonl行号': line_num + 1  # 显示为1-based
                        })
                        
            except json.JSONDecodeError as e:
                print(f"错误: 预测文件第{line_num+1}行JSON解析错误: {e}")
                all_actions.append({
                    'BOS路径': f'错误_行{line_num+1}',
                    '预测标签': 'JSON解析错误',
                    '真实标签': '',
                    '是否一致': '',
                    '开始时间': '',
                    '结束时间': '',
                    '备注': f'JSON解析错误: {str(e)[:50]}',
                    '原predict': line[:100] if len(line) > 100 else line,
                    'jsonl行号': line_num + 1
                })
                continue
            except Exception as e:
                print(f"错误: 预测文件第{line_num+1}行处理出错: {e}")
                all_actions.append({
                    'BOS路径': f'错误_行{line_num+1}',
                    '预测标签': '处理错误',
                    '真实标签': '',
                    '是否一致': '',
                    '开始时间': '',
                    '结束时间': '',
                    '备注': f'处理错误: {str(e)[:50]}',
                    '原predict': 'N/A',
                    'jsonl行号': line_num + 1
                })
                continue
    
    # 检查行数是否匹配
    if line_num + 1 != len(video_paths):
        print(f"\n警告: 行数不匹配!")
        print(f"预测结果文件行数: {line_num + 1}")
        print(f"原始数据集样本数: {len(video_paths)}")
        print("这可能意味着预测结果和原始数据集不完全对应，请检查数据!")
    
    if not all_actions:
        print("未找到任何动作信息")
        return None
    
    # 3. 创建DataFrame
    df = pd.DataFrame(all_actions)
    
    # 4. 保存到Excel
    with pd.ExcelWriter(output_excel_path, engine='openpyxl') as writer:
        # 主表
        df.to_excel(writer, sheet_name='验证结果', index=False)
        
        # 设置列宽
        worksheet = writer.sheets['验证结果']
        column_widths = {
            'A': 50,  # BOS路径
            'B': 30,  # 预测标签
            'C': 30,  # 真实标签
            'D': 12,  # 是否一致
            'E': 12,  # 开始时间
            'F': 12,  # 结束时间
            'G': 20,  # 备注
            'H': 60,  # 原predict
            'I': 10,  # jsonl行号
        }
        
        for col_letter, width in column_widths.items():
            worksheet.column_dimensions[col_letter].width = width
        
        # 添加汇总表
        summary_data = {
            '统计项': ['总样本数', '总动作数', '视频路径有效数', '已验证数', '一致数', '不一致数', '一致率'],
            '数量/比例': [
                line_num + 1,
                len([a for a in all_actions if a['预测标签'] not in ['无预测结果', 'JSON解析错误', '处理错误']]),
                len([a for a in all_actions if a['BOS路径'] != 'N/A' and not a['BOS路径'].startswith('错误')]),
                0, 0, 0, '0%'
            ]
        }
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='汇总', index=False)
        
        # 添加标签统计
        valid_predictions = df[~df['预测标签'].isin(['无预测结果', 'JSON解析错误', '处理错误'])]
        if not valid_predictions.empty:
            label_counts = valid_predictions['预测标签'].value_counts().reset_index()
            label_counts.columns = ['预测标签', '出现次数']
            label_counts.to_excel(writer, sheet_name='标签统计', index=False)
        
        # 添加错误统计
        error_types = df[df['预测标签'].isin(['无预测结果', 'JSON解析错误', '处理错误'])]
        if not error_types.empty:
            error_counts = error_types['预测标签'].value_counts().reset_index()
            error_counts.columns = ['错误类型', '出现次数']
            error_counts.to_excel(writer, sheet_name='错误统计', index=False)
    
    print(f"\n{'='*50}")
    print(f"成功生成Excel文件: {output_excel_path}")
    print(f"总计处理了 {line_num + 1} 个预测样本")
    print(f"提取了 {len([a for a in all_actions if a['预测标签'] not in ['无预测结果', 'JSON解析错误', '处理错误']])} 个有效动作")
    print(f"Excel包含以下工作表:")
    print("1. 验证结果 - 包含所有动作信息和视频路径")
    print("2. 汇总 - 统计信息")
    print("3. 标签统计 - 各标签出现次数")
    if not error_types.empty:
        print("4. 错误统计 - 错误类型统计")
    
    return output_excel_path

# 使用示例
if __name__ == "__main__":
    # 文件路径
    predict_jsonl_path = "/root/workspace/LLaMA-Factory/infer_results/Qwen3-VL-4B-digged_dataset.jsonl"
    dataset_json_path = "/root/workspace/LLaMA-Factory/data/digged_dataset.json"
    
    # 检查文件是否存在
    if not os.path.exists(predict_jsonl_path):
        print(f"错误: 预测结果文件不存在: {predict_jsonl_path}")
    elif not os.path.exists(dataset_json_path):
        print(f"错误: 原始数据集文件不存在: {dataset_json_path}")
    else:
        # 生成带视频路径的验证Excel
        excel_path = generate_validation_excel_with_video_paths(
            predict_jsonl_path, 
            dataset_json_path, 
            "validation_with_video_paths.xlsx"
        )
        
        if excel_path:
            print(f"\n{'='*50}")
            print("Excel文件已生成!")
            print(f"文件路径: {excel_path}")
            print("\n使用说明:")
            print("1. 打开Excel文件，查看'验证结果'工作表")
            print("2. 在'是否一致'列填写: ✓(正确) 或 ✗(错误)")
            print("3. 如果不一致，在'备注'列填写原因")
            print("4. 视频路径可以直接点击查看或复制")