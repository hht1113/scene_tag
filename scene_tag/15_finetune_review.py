#!/usr/bin/env python3
"""
精调验证人工抽查工具

用法:
    python scene_tag/15_finetune_review.py --port 7862
"""

import argparse
import json
import os
import glob
from pathlib import Path
from PIL import Image

try:
    import gradio as gr
except ImportError:
    print("请安装 gradio: pip install gradio")
    exit(1)

REVIEW_DIRS = [
    "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/finetune_review",
    "/root/workspace/LLaMA-Factory/scene_tag/results_ab_compare/generalize_review",
]
MAX_IMG_WIDTH = 960


def _find_file(filename):
    for d in REVIEW_DIRS:
        fp = os.path.join(d, filename)
        if os.path.exists(fp):
            return fp
    return None


def list_review_files():
    files = []
    for d in REVIEW_DIRS:
        files.extend(sorted(glob.glob(os.path.join(d, "*.json"))))
    return [os.path.basename(f) for f in files]


def load_review_file(filename):
    fp = _find_file(filename)
    if not fp:
        raise FileNotFoundError(f"未找到: {filename}")
    with open(fp, "r", encoding="utf-8") as f:
        return json.load(f)


def save_review_file(filename, data):
    fp = _find_file(filename)
    if not fp:
        fp = os.path.join(REVIEW_DIRS[0], filename)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_image_resized(path):
    if not os.path.exists(path):
        return None
    try:
        img = Image.open(path)
        if img.width > MAX_IMG_WIDTH:
            ratio = MAX_IMG_WIDTH / img.width
            img = img.resize((MAX_IMG_WIDTH, int(img.height * ratio)), Image.LANCZOS)
        return img
    except Exception:
        return None


current_data = []
current_file = ""
current_idx = 0


def get_stats_str():
    if not current_data:
        return ""
    reviewed = sum(1 for d in current_data if d.get("人工判定"))
    correct = sum(1 for d in current_data if d.get("人工判定") == "正确")
    total = len(current_data)
    if reviewed == 0:
        return f"待审 {total} 条"
    return f"已审 {reviewed}/{total}，正确 {correct}，Precision = {correct}/{reviewed} = {correct/reviewed*100:.0f}%"


def get_item_display():
    if not current_data:
        return None, "无数据", None, "", f"0/0"

    item = current_data[current_idx]
    img = load_image_resized(item["图片路径"])

    labels = item.get("模型判定", {})
    true_labels = [k for k, v in labels.items() if v]
    false_labels = [k for k, v in labels.items() if not v]
    label_text = f"✅ 命中: {', '.join(true_labels) if true_labels else '无'}\n❌ 未命中: {', '.join(false_labels)}"

    judgment = item.get("人工判定") or None
    note = item.get("备注", "")
    progress = f"第 {current_idx + 1} / {len(current_data)} 条"

    return img, label_text, judgment, note, progress


def select_file(filename):
    global current_data, current_file, current_idx
    if not filename:
        return None, "", None, "", "0/0", ""
    current_file = filename
    current_data = load_review_file(filename)
    current_idx = 0
    img, labels, j, n, prog = get_item_display()
    return img, labels, j, n, prog, get_stats_str()


def go_prev():
    global current_idx
    if current_data:
        current_idx = max(0, current_idx - 1)
    return get_item_display()


def go_next():
    global current_idx
    if current_data:
        current_idx = min(len(current_data) - 1, current_idx + 1)
    return get_item_display()


def submit_judgment(judgment, note):
    global current_idx
    if not current_data or not judgment:
        return get_item_display() + (get_stats_str(),)

    current_data[current_idx]["人工判定"] = judgment
    current_data[current_idx]["备注"] = note or ""
    save_review_file(current_file, current_data)

    if current_idx < len(current_data) - 1:
        current_idx += 1

    img, labels, j, n, prog = get_item_display()
    return img, labels, None, "", prog, get_stats_str()


def calc_stats():
    lines = []
    for fname in list_review_files():
        data = load_review_file(fname)
        total = len(data)
        reviewed = sum(1 for d in data if d.get("人工判定"))
        correct = sum(1 for d in data if d.get("人工判定") == "正确")
        if reviewed > 0:
            lines.append(f"{fname}:  {correct}/{reviewed} = {correct/reviewed*100:.0f}%")
        else:
            lines.append(f"{fname}:  未开始 ({total}条)")
    return "\n".join(lines)


def build_ui():
    with gr.Blocks(title="精调验证人工抽查") as demo:
        gr.Markdown("# 精调验证 - 人工抽查工具\n选择文件 → 看图 → 点「正确」或「错误」→ 自动保存并跳下一条")

        with gr.Row():
            file_dropdown = gr.Dropdown(choices=list_review_files(), label="选择抽查文件", scale=3)
            stats_text = gr.Textbox(label="统计", scale=2, interactive=False)

        with gr.Row():
            with gr.Column(scale=3):
                image_display = gr.Image(label="图片", height=500, type="pil")
            with gr.Column(scale=2):
                progress_text = gr.Textbox(label="进度", interactive=False)
                label_display = gr.Textbox(label="模型判定", lines=3, interactive=False)
                judgment_input = gr.Radio(choices=["正确", "错误"], label="人工判定", value=None)
                note_input = gr.Textbox(label="备注（可选）", lines=1)

                with gr.Row():
                    prev_btn = gr.Button("⬅ 上一条")
                    submit_btn = gr.Button("提交并下一条 ➡", variant="primary")
                    next_btn = gr.Button("跳过 ➡")

        with gr.Accordion("全部统计", open=False):
            all_stats_btn = gr.Button("刷新统计")
            all_stats_text = gr.Textbox(label="全部文件 Precision", lines=15, interactive=False)

        outputs_5 = [image_display, label_display, judgment_input, note_input, progress_text]
        outputs_6 = outputs_5 + [stats_text]

        file_dropdown.change(fn=select_file, inputs=[file_dropdown], outputs=outputs_6)
        prev_btn.click(fn=go_prev, outputs=outputs_5)
        next_btn.click(fn=go_next, outputs=outputs_5)
        submit_btn.click(fn=submit_judgment, inputs=[judgment_input, note_input], outputs=outputs_6)
        all_stats_btn.click(fn=calc_stats, outputs=[all_stats_text])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7862)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=True,
        allowed_paths=["/mnt/pfs/", "/root/workspace/"],
    )
