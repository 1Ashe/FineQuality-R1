
import re, os
from datetime import datetime
from pathlib import Path
import PIL.Image
from PIL import Image
from itertools import combinations
import itertools
from datetime import datetime
import numpy as np 
import pathlib
import random
from datetime import datetime
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from babel.numbers import parse_decimal
from utils.math import compute_score
from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import VLMGRPOTrainer, GRPOConfig
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
import PIL
from Levenshtein import ratio
from open_r1.utils.pycocotools.coco import COCO
from open_r1.utils.pycocotools.cocoeval import COCOeval
import json
import math
from json_repair import repair_json

from open_r1.vlm_modules import *

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionFlashAttention2, apply_rotary_pos_emb_flashatt, flash_attn_varlen_func
import torch
from typing import Tuple
from transformers.utils import logging
from transformers import AutoProcessor, AutoTokenizer

from openai import OpenAI
import re, os
from datetime import datetime
from pathlib import Path
import PIL.Image

logger = logging.get_logger(__name__)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY", "<OPENAI_API_KEY>"),
    base_url=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
)

from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
monkey_patch_qwen2_5vl_flash_attn()    

tokenizer = None

def initialize_tokenizer(model_path):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    return tokenizer

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.
    """
    data_file_paths: str = field(
        default=None,
        metadata={"help": "Paths to data files, separated by ':'"},
    )
    image_folders: str = field(
        default=None,
        metadata={"help": "Paths to image folders, separated by ':'"},
    )
    gray_image_folders: str = field(
        default=None,
        metadata={"help": "Paths to gray_image folders, separated by ':'"},
    )
    arrow_cache_dir: str = field(
        default=None,
        metadata={"help": "Path to arrow cache directory"},
    )
    val_split_ratio: float = field(
        default=0.0,
        metadata={"help": "Ratio of validation split, default 0.0"},
    )
    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    reward_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "Choose reward method: 'default', 'mcp', ..."
        },
    )
    question_template: Optional[str] = field(
        default="scoring",
        metadata={
            "help": "Choose scoring or comparing question"
        },
    )


def extract_first_number(model_answer):
    match = re.search(r'-?\d+(\.\d+)?', model_answer)
    if match:
        return float(match.group())
    else:
        return random.randint(1, 5)

def fidelity_reward(pred1, pred2, var1, var2, gt, device):
    esp = 1e-6
    try:
        normal_dist = torch.distributions.Normal(0, 1)
        _cur = (pred1 - pred2) / torch.sqrt(var1 + var2 + esp)
        p = normal_dist.cdf(_cur)
    except:
        print("Meet Error ...")
        p = torch.tensor(0.5, dtype=torch.float32, device=device)
    
    reward = torch.sqrt(p * gt + esp) + torch.sqrt((1 - p) * (1 - gt) + esp)
    return reward

def accuracy_reward(completions, solution, **kwargs):
    """
    Reward function that checks if the completion is correct using symbolic verification, 
    exact string matching, or fuzzy matching. Also tracks prediction errors for error_matrix.
    """
    device = kwargs.get("device")
    n_gen = kwargs.get("num_generations")
    
    sample_ids_raw = [completion[0].get("id", None) for completion in completions]
    sample_ids = [sample_ids_raw[i] for i in range(0, len(sample_ids_raw), n_gen)]
    
    current_epoch_errors = kwargs.get("current_epoch_errors", None)
    
    reshaped_solution = [solution[i:i + n_gen] for i in range(0, len(solution), n_gen)]
    for i in range(len(reshaped_solution)):
        for j in range(len(reshaped_solution[i])):
            _cur = reshaped_solution[i][j]
            sol_match = re.search(r'<answer>(.*?)</answer>', _cur)
            ground_truth = sol_match.group(1).strip() if sol_match else _cur.strip()
            reshaped_solution[i][j] = float(ground_truth)

    contents = [completion[0]["content"] for completion in completions]
    reshaped_content = [contents[i:i + n_gen] for i in range(0, len(contents), n_gen)]

    batch_mean, batch_var, batch_pred = [], [], []
    
    for i in range(len(reshaped_content)): 
        cur_pred_list = []
        for j in range(len(reshaped_content[i])): 
            try:
                content_matches = re.findall(r'<answer>(.*?)</answer>', reshaped_content[i][j], re.DOTALL)
                student_answer = content_matches[-1].strip() if content_matches else reshaped_content[i][j].strip()
                pred = extract_first_number(student_answer)
            except:
                print("Meet Error ...")
                pred = random.uniform(1, 5)
            cur_pred_list.append(pred)
        
        batch_pred.append(cur_pred_list)
        p = torch.tensor(cur_pred_list, dtype=torch.float32, device=device)
        p_mean = torch.mean(p)
        p_var = torch.var(p)
        batch_mean.append([p_mean])
        batch_var.append([p_var])
    
    rewards = []
    
    num_samples_in_batch = len(batch_pred)
    
    for i in range(len(batch_pred)):
        for j in range(len(batch_pred[i])):
            _reward_sum, _count_idx = 0, 0
            
            for z in range(len(batch_mean)):
                if z != i:
                    input_pred1 = batch_pred[i][j]
                    input_pred2 = batch_mean[z][0]
                    input_var1 = batch_var[i][0]
                    input_var2 = batch_var[z][0]

                    if reshaped_solution[i][j] > reshaped_solution[z][0]:
                        input_gt = torch.tensor(1.0, dtype=torch.float32, device=device)
                        gt_comparison = 1
                    elif reshaped_solution[i][j] < reshaped_solution[z][0]:
                        input_gt = torch.tensor(0.0, dtype=torch.float32, device=device)
                        gt_comparison = -1
                    else:
                        input_gt = torch.tensor(0.5, dtype=torch.float32, device=device)
                        gt_comparison = 0

                    _reward = fidelity_reward(
                        pred1=input_pred1, pred2=input_pred2, var1=input_var1, 
                        var2=input_var2, gt=input_gt, device=device
                    )

                    if gt_comparison != 0:
                        pred_comparison = 1 if input_pred1 > input_pred2 else (-1 if input_pred1 < input_pred2 else 0)
                        
                        is_prediction_error = (pred_comparison != gt_comparison)
                        
                        if (is_prediction_error and 
                            sample_ids is not None and 
                            current_epoch_errors is not None):
                            
                            global_idx_i = sample_ids[i] - 1
                            global_idx_z = sample_ids[z] - 1
                            
                            current_epoch_errors[global_idx_i, global_idx_z] += 1

                    _reward_sum = _reward_sum + _reward
                    _count_idx = _count_idx + 1

            _cur_reward = _reward_sum / _count_idx
            rewards.append(_cur_reward)

            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                image_path = kwargs.get("image_path") if "image_path" in kwargs else None
                problem = kwargs.get("problem")[0]
                image_path = [image_path[i:i + n_gen] for i in range(0, len(image_path), n_gen)]

                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"\n------------- {current_time} Accuracy reward: {_cur_reward} -------------\n")
                    f.write(f"accu_reward_method: {_cur_reward}\n")
                    f.write(f"image_path: {image_path[i][j]}\n")
                    f.write(f"problem: {problem}\n")
                    f.write(f"Content: {reshaped_content[i][j]}\n")
                    f.write(f"Solution: {reshaped_solution[i][j]}\n\n") 

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]

    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    if os.getenv("DEBUG_MODE") == "true":
        log_path = os.getenv("LOG_PATH")
        with open(log_path.replace(".txt", "_format.txt"), "a", encoding='utf-8') as f:
            f.write(f"------------- {current_time} Format reward -------------\n")
            for content, match in zip(completion_contents, matches):
                f.write(f"Content: {content}\n")
                f.write(f"Has format: {bool(match)}\n")

    return [1.0 if match else 0.0 for match in matches]



def format_reward(completions, **kwargs):
    images = kwargs.get("images")
    gray_images = kwargs.get("gray_images")
    image_paths = kwargs.get("image_path", [])

    coord_pattern = re.compile(r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]")
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    log_path = os.getenv("LOG_PATH", "reward_log.txt")

    BASE_FORMAT_REWARD = 0.2
    BASE_ANSWER_REWARD = 0.2
    BASE_COORD_REWARD  = 0.2

    MIN_AREA_RATIO = 0.05
    MAX_AREA_RATIO = 0.60
    HIGH_IOU_THRESHOLD = 0.07
    SEVERE_MULTIPLIER = 0.5
    EXCESS_PENALTY = 0.1  

    def iou(b1, b2):
        x1, y1, x2, y2 = b1
        x3, y3, x4, y4 = b2
        ix1, iy1 = max(x1, x3), max(y1, y3)
        ix2, iy2 = min(x2, x4), min(y2, y4)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        a1 = (x2 - x1) * (y2 - y1)
        a2 = (x4 - x3) * (y4 - y3)
        union = a1 + a2 - inter
        return 0.0 if union == 0 else inter / union

    def is_contained(inner, outer):
        return (
            inner[0] >= outer[0] and
            inner[1] >= outer[1] and
            inner[2] <= outer[2] and
            inner[3] <= outer[3]
        )

    def get_image_size(img):
        if isinstance(img, PIL.Image.Image):
            return img.size
        if isinstance(img, (str, Path)) and Path(img).exists():
            with PIL.Image.open(img) as im:
                return im.size
        return None

    def is_valid_answer_score(text):
        try:
            v = float(text.strip())
            return 1.0 <= v <= 5.0
        except:
            return False

    rewards = []
    completion_details = []

    for idx, completion in enumerate(completions):
        content = completion[0]["content"]
        reward = 0.0
        logs = []

        img_w = img_h = img_area = None
        if images and idx < len(images):
            size = get_image_size(images[idx])
            if size:
                img_w, img_h = size
                img_area = img_w * img_h

        has_tags = all([
            re.search(r"<think\s*>", content, re.I),
            re.search(r"</think\s*>", content, re.I),
            re.search(r"<answer\s*>", content, re.I),
            re.search(r"</answer\s*>", content, re.I),
        ])

        valid_coords = []
        coord_score_sum = 0.0

        if has_tags:
            reward += BASE_FORMAT_REWARD
            logs.append(f"格式标签对 +{BASE_FORMAT_REWARD}")

            m = re.search(r"<answer\s*>(.*?)</answer\s*>", content, re.S | re.I)
            if m and is_valid_answer_score(m.group(1)):
                reward += BASE_ANSWER_REWARD
                logs.append(f"答案合法 +{BASE_ANSWER_REWARD}")
        else:
            logs.append("缺少格式标签")

        think_m = re.search(r"<think\s*>(.*?)</think\s*>", content, re.S | re.I)
        if think_m and img_w and img_h:
            think_content = think_m.group(1)
            sentences = re.split(r'(?<=[.])', think_content)

            for sent in sentences:
                for m in coord_pattern.finditer(sent):
                    x1, y1, x2, y2 = map(int, m.groups())
                    box = (x1, y1, x2, y2)
                    coord_str = str(box)

                    if (x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h or x1 >= x2 or y1 >= y2):
                        logs.append(f"{coord_str} 越界或非法 → 跳过")
                        continue
                    
                    if any(is_contained(box, v["box"]) for v in valid_coords):
                        logs.append(f"{coord_str} 被已有框包含 → 跳过")
                        continue

                    multiplier = 1.0
                    penalties = []

                    if any(iou(box, v["box"]) > HIGH_IOU_THRESHOLD for v in valid_coords):
                        multiplier *= SEVERE_MULTIPLIER
                        penalties.append("高IoU")

                    box_area = (x2 - x1) * (y2 - y1)
                    area_ratio = box_area / img_area
                    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
                        multiplier *= SEVERE_MULTIPLIER
                        penalties.append("面积比例异常")

                    curr_coord_reward = BASE_COORD_REWARD * multiplier
                    
                    valid_coords.append({
                        "box": box,
                        "reward": curr_coord_reward,
                        "sentence": sent
                    })

                    if len(valid_coords) <= 3:
                        coord_score_sum += curr_coord_reward
                        logs.append(f"坐标#{len(valid_coords)} {coord_str}: +{curr_coord_reward:.2f} " + (f"[{','.join(penalties)}]" if penalties else ""))
                    else:
                        logs.append(f"坐标#{len(valid_coords)} {coord_str}: 超过3个不计分")

            if len(valid_coords) > 3:
                coord_score_sum -= EXCESS_PENALTY
                logs.append(f"冗余惩罚 (总数{len(valid_coords)}): -{EXCESS_PENALTY}")

            reward += coord_score_sum

        reward = round(max(0.0, reward), 4)
        
        completion_details.append({
            "content": content,
            "valid_coords": valid_coords,
            "format_reward": reward,
            "log_messages": logs
        })

    saliency_rewards = []
    saliency_log_details = []
    for idx, detail in enumerate(completion_details):
        gray = gray_images[idx] if gray_images and idx < len(gray_images) else None
        s_reward, s_logs = saliency_reward(detail["valid_coords"], gray)
        saliency_rewards.append(s_reward)
        saliency_log_details.append(s_logs)

    final_rewards = []
    for idx, (detail, s_reward) in enumerate(zip(completion_details, saliency_rewards)):
        final_score = round((detail["format_reward"] + s_reward) * 0.25, 4)
        final_rewards.append(final_score)

    if debug_mode:
        log_file = log_path.replace(".txt", "_format.txt")
        for idx in range(len(final_rewards)):
            detail = completion_details[idx]
            f_reward = detail['format_reward']
            s_reward = saliency_rewards[idx]
            total = final_rewards[idx]

            with open(log_file, "a", encoding="utf-8") as f_log:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f_log.write(f"\n========== Index {idx} | {current_time} ==========\n")

                if image_paths and idx < len(image_paths):
                    f_log.write(f"图片路径: {image_paths[idx]}\n")

                f_log.write(f"输出:\n{completions[idx][0]['content']}\n")

                f_log.write("\n---- Format Reward ----\n")
                for msg in detail['log_messages']:
                    f_log.write(f"{msg}\n")

                f_log.write("\n---- Saliency Reward ----\n")
                for msg in saliency_log_details[idx]:
                    f_log.write(f"{msg}\n")

                f_log.write("\n==== Total Reward ====\n")
                f_log.write(f"Format Reward: {f_reward:.4f} | Saliency Reward: {s_reward:.4f} \n")
                f_log.write(f"Total Reward: {total:.4f}\n")
                f_log.write("="*80 + "\n\n")

    return final_rewards


def saliency_reward(valid_coords, gray_img):
    SALIENCY_MAX_REWARD = 0.30
    VAR_MAX = 0.25
    VAR_MAX_REWARD = 0.8

    log_messages = []
    if not valid_coords or gray_img is None:
        log_messages.append("缺少有效坐标或灰度图, 奖励=0.0")
        return 0.0, log_messages

    def _normalize_gray(img):
        if img.mode != 'L': img = img.convert('L')
        arr = np.array(img, dtype=np.float32)
        min_val, max_val = arr.min(), arr.max()
        return (arr - min_val) / (max_val - min_val) if max_val > min_val else np.zeros_like(arr)

    def _compute_box_variance_reward(coord, norm_gray, width, height):
        x1, y1, x2, y2 = coord['box']
        x1, y1 = int(max(0, x1)), int(max(0, y1))
        x2, y2 = int(min(width, x2)), int(min(height, y2))
        box_pixels = norm_gray[y1:y2, x1:x2]
        if box_pixels.size == 0: return 0.0, None, 0.0
        var = np.var(box_pixels)
        var_reward = max(0.0, VAR_MAX - var) * VAR_MAX_REWARD
        mask = np.zeros((height, width), dtype=bool)
        mask[y1:y2, x1:x2] = True
        return var_reward, mask, var

    width, height = gray_img.size
    norm_gray = _normalize_gray(gray_img)
    
    masks, var_rewards = [], []
    for idx, coord in enumerate(valid_coords):
        v_rew, mask, var = _compute_box_variance_reward(coord, norm_gray, width, height)
        if mask is not None: masks.append(mask)
        var_rewards.append(v_rew)
        log_messages.append(f"坐标#{idx+1}: {coord['box']}, 方差={var:.4f}, 奖励={v_rew:.4f}")

    mean_var_reward = float(np.mean(var_rewards)) if var_rewards else 0.0
    
    if masks:
        combined_mask = np.any(masks, axis=0)
        coverage = norm_gray[combined_mask].sum() / norm_gray.sum() if norm_gray.sum() > 0 else 0.0
    else:
        coverage = 0.0
        
    coverage_reward = coverage * SALIENCY_MAX_REWARD
    final_s_reward = mean_var_reward + coverage_reward
    
    log_messages.append(f"覆盖率: {coverage:.4f}, 覆盖奖励: {coverage_reward:.4f}, 最终显著性得分: {final_s_reward:.4f}")
    return final_s_reward, log_messages


"""
重复性奖励
"""
def repetitive_reward(completions, **kwargs):

    completion_ids = kwargs.get("completion_ids", [])
    pad_token_id = kwargs.get("pad_token_id")
    image_paths = kwargs.get("image_paths")
    
    MAX_REWARD = 0.5
    N_GRAM = 8
    
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    log_path = os.getenv("LOG_PATH", "reward_log.txt")
    
    rewards = []
    reward_details = []
    
    for idx, (completion, ids) in enumerate(zip(completions, completion_ids)):
        content = completion[0]["content"]
        text = content
        log_messages = []
        
        if not text.strip():
            rewards.append(MAX_REWARD)
            log_messages.append(f"文本为空,给予最大奖励: {MAX_REWARD}")
            reward_details.append({'log_messages': log_messages, 'reward': MAX_REWARD})
            continue
        
        words = text.split()
        if len(words) < 2 * N_GRAM:
            rewards.append(MAX_REWARD)
            log_messages.append(f"单词数({len(words)})<{2*N_GRAM},给予最大奖励: {MAX_REWARD}")
            reward_details.append({'log_messages': log_messages, 'reward': MAX_REWARD})
            continue
        
        repeat_count = sum(1 for i in range(len(words) - 2*N_GRAM + 1) 
                          if words[i:i+N_GRAM] == words[i+N_GRAM:i+2*N_GRAM])
        text_score = 1.0 - repeat_count / (len(words) - 2*N_GRAM + 1)
        log_messages.append(f"文本重复: {repeat_count}次/{len(words) - 2*N_GRAM + 1}, 得分={text_score:.4f}")
        
        ids = ids.tolist() if hasattr(ids, 'tolist') else ids
        if pad_token_id in ids:
            ids = ids[:ids.index(pad_token_id)]
       
        if len(ids) < 2 * N_GRAM:
            token_score = 1.0
            log_messages.append(f"Token数({len(ids)})<{2*N_GRAM}, token得分=1.0")
        else:
            repeat_count = sum(1 for i in range(len(ids) - 2*N_GRAM + 1) 
                             if ids[i:i+N_GRAM] == ids[i+N_GRAM:i+2*N_GRAM])
            token_score = 1.0 - repeat_count / (len(ids) - 2*N_GRAM + 1)
            log_messages.append(f"Token重复: {repeat_count}次/{len(ids) - 2*N_GRAM + 1}, 得分={token_score:.4f}")
        
        score = (text_score + token_score) / 2
        reward = round(score * MAX_REWARD, 4)
        log_messages.append(f"平均得分={(text_score + token_score)/2:.4f}, 最终奖励={reward:.4f}")
        
        rewards.append(reward)
        reward_details.append({'log_messages': log_messages, 'reward': reward})
    
    if debug_mode:
        log_file = log_path.replace(".txt", "_repetitive.txt")
        for idx, detail in enumerate(reward_details):
            with open(log_file, "a", encoding="utf-8") as f_log:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f_log.write(f"========== Repetitive Reward | Index {idx} | {current_time} ==========\n")
                if image_paths and idx < len(image_paths):
                    f_log.write(f"彩色图: {image_paths[idx]}\n")
                f_log.write(f"输出:\n{completions[idx][0]['content']}\n")
                f_log.write(f"--- 详细评分 ---\n")
                for msg in detail['log_messages']:
                    f_log.write(f"{msg}\n")
                f_log.write(f"Repetitive总分: {detail['reward']:.4f}\n")
                f_log.write("="*80 + "\n\n")
    
    return rewards

"""
丰富度奖励
"""
def richness_reward(completions, **kwargs):

    image_paths = kwargs.get("image_paths")
    
    MAX_REWARD = 0.5
    WORD_LOW = 45
    WORD_HIGH = 65
    PENALTY_STEP = 0.02
    
    debug_mode = os.getenv("DEBUG_MODE") == "true"
    log_path = os.getenv("LOG_PATH", "reward_log.txt")
    
    coord_pat = re.compile(r"\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]")
    rewards = []
    reward_details = []
   
    for idx, comp in enumerate(completions):
        content = comp[0]["content"]
        log_messages = []
        
        m = re.search(r"<think\s*>(.*?)</think\s*>", content, re.DOTALL | re.IGNORECASE)
        if not m:
            rewards.append(0.0)
            log_messages.append("未找到<think>标签, 奖励=0.0")
            reward_details.append({'log_messages': log_messages, 'reward': 0.0})
            continue
        
        text = coord_pat.sub(" ", m.group(1))
        words = [w for w in re.findall(r'\b[A-Za-z0-9]+\b', text) if re.search(r'[A-Za-z]', w)]
        wc = len(words)
        log_messages.append(f"有效单词数: {wc}")
        
        if wc <= WORD_LOW:
            r = wc / WORD_LOW
            log_messages.append(f"单词数≤{WORD_LOW}, 得分={r:.4f}")
        elif wc <= WORD_HIGH:
            r = 1.0
            log_messages.append(f"单词数在[{WORD_LOW},{WORD_HIGH}]区间, 得分=1.0")
        else:
            excess = wc - WORD_HIGH
            r = max(0.0, 1.0 - excess * PENALTY_STEP)
            log_messages.append(f"单词数>{WORD_HIGH}, 超出{excess}个, 惩罚={excess * PENALTY_STEP:.4f}, 得分={r:.4f}")
        
        reward = round(r * MAX_REWARD, 4)
        log_messages.append(f"最终奖励={reward:.4f}")
        
        rewards.append(reward)
        reward_details.append({'log_messages': log_messages, 'reward': reward})
    
    if debug_mode:
        log_file = log_path.replace(".txt", "_richness.txt")
        for idx, detail in enumerate(reward_details):
            with open(log_file, "a", encoding="utf-8") as f_log:
                current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
                f_log.write(f"========== Richness Reward | Index {idx} | {current_time} ==========\n")
                if image_paths and idx < len(image_paths):
                    f_log.write(f"彩色图: {image_paths[idx]}\n")
                f_log.write(f"输出:\n{completions[idx][0]['content']}\n")
                f_log.write(f"--- 详细评分 ---\n")
                for msg in detail['log_messages']:
                    f_log.write(f"{msg}\n")
                f_log.write(f"Richness总分: {detail['reward']:.4f}\n")
                f_log.write("="*80 + "\n\n")
    
    return rewards

reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "repetitive": repetitive_reward,
    "richness": richness_reward
}

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


def get_vlm_module(model_name_or_path):
    return Qwen2VLModule

def main(script_args, training_args, model_args):
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)
    print("using vlm module:", vlm_module_cls.__name__)
    question_prompt = vlm_module_cls.get_question_template(task_type=script_args.question_template)

    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    import json
    from datasets import Dataset
    
    data_files = script_args.data_file_paths.split(":")
    image_folders = script_args.image_folders.split(":")
    gray_image_folders = script_args.gray_image_folders.split(":")

    training_args.max_completion_length = 256

    if len(data_files) != len(image_folders):
        raise ValueError("Number of data files must match number of image folders")

    if len(gray_image_folders) != len(data_files):
        raise ValueError("Number of gray_image folders must match number of data files")

    if script_args.reward_method is None:
        accu_reward_methods = ["default"] * len(data_files)
    else:
        accu_reward_methods = script_args.reward_method.split(":")
        assert len(accu_reward_methods) == len(data_files), f"Number of reward methods must match number of data files: {len(accu_reward_methods)} != {len(data_files)}"

    all_data = []
    values = []
    for data_file, image_folder, gray_image_folder, accu_reward_method in zip(data_files, image_folders, gray_image_folders, accu_reward_methods):
        with open(data_file, 'r') as f:
            for line in f:
                item = json.loads(line)
                row_gpt_values = [conv['value'] for conv in item.get('conversations', []) if conv.get('from') == 'gpt']
                values.extend(row_gpt_values)

                if 'image' in item:
                    if isinstance(item['image'], str):
                        item['image_path'] = [os.path.join(image_folder, item['image'])]
                        item['gray_image_path'] = [os.path.join(gray_image_folder, item['image'])]
                        del item['image']
                    elif isinstance(item['image'], list):
                        item['image_path'] = [os.path.join(image_folder, image) for image in item['image']]
                        item['gray_image_path'] = [os.path.join(gray_image_folder, image) for image in item['image']]
                        del item['image']
                    else:
                        raise ValueError(f"Unsupported image type: {type(item['image'])}")
                
                item['problem'] = item['conversations'][0]['value'].replace('<image>', '')
                
                solution_value = item['conversations'][1]['value']
                if isinstance(solution_value, str):
                    item['solution'] = solution_value.replace('<answer>', '').replace('</answer>', '').strip()
                else:
                    item['solution'] = str(solution_value)
                
                del item['conversations']
                item['accu_reward_method'] = item.get('accu_reward_method', accu_reward_method)
                all_data.append(item)

    dataset = Dataset.from_list(all_data)

    def make_conversation_from_jsonl(example):
        if 'image_path' in example and example['image_path'] is not None:
            return {
                'image_path': [p for p in example['image_path']],
                'dataset_name': example['dataset_name'],
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        *({'type': 'image', 'text': None} for _ in range(len(example['image_path']))),
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }
        else:
            return {
                'dataset_name': example['dataset_name'],
                'problem': example['problem'],
                'solution': f"<answer> {example['solution']} </answer>",
                'accu_reward_method': example['accu_reward_method'],
                'prompt': [{
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': question_prompt.format(Question=example['problem'])}
                    ]
                }]
            }

    dataset = dataset.map(make_conversation_from_jsonl, num_proc=8)

    splits = {'train': dataset}
    if script_args.val_split_ratio > 0:
        train_val_split = dataset.train_test_split(
            test_size=script_args.val_split_ratio
        )
        splits['train'] = train_val_split['train']
        splits['validation'] = train_val_split['test']

    trainer_cls = VLMGRPOTrainer
    print("using trainer:", trainer_cls.__name__)
    initialize_tokenizer(model_args.model_name_or_path)

    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=splits['train'],
        eval_dataset=splits.get('validation') if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        values = values,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub()


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    if training_args.deepspeed and "zero3" in training_args.deepspeed:
        print("zero3 is used, qwen2_5vl forward monkey patch is applied")
        monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
