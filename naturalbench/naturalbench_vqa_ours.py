import torch
import re
from tqdm import tqdm
import json
import os
from typing import List, Optional, Union, Dict, Any, Callable, Tuple

from datasets import load_dataset
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from transformers.generation.utils import GenerationConfig, GenerationMixin, LogitsProcessorList
from transformers.generation.stopping_criteria import StoppingCriteriaList
from qwen_vl_utils import process_vision_info  # pip install qwen-vl-utils[decord]

# from custom_utils import GenerationMixin

SUFFIX_FOR_VQA = {
    "yes_no": "Please answer Yes or No.",
    "multiple_choice": "Please output the letter corresponding to the correct option."
} 

def custom_manipulation_fn(hidden_states, step_idx=None):
    """
    Custom manipulation function for the hidden states.
    
    Args:
        hidden_states (torch.Tensor): The hidden states to manipulate.
            Shape (batch_size, seq_len, hidden_dim)
        step_idx (int, optional): The current generation step. None if not in step-by-step generation.
        
    Returns:
        torch.Tensor: The manipulated hidden states.
    """
    # 여기에 hidden states 조작 로직 구현
    # 예: 특정 차원의 값을 증폭하거나 감소시킴
    # 토큰 생성 단계(step_idx)에 따라 다른 조작을 적용할 수도 있음
    
    # 예시: 단순히 hidden states의 특정 차원을 증폭
    if step_idx is not None:
        # 특정 차원 증폭 (예: 첫 10개 차원)
        scaling_factor = 1.0 + (0.1 * (step_idx % 5))  # 스텝마다 다른 스케일링
        hidden_states[:, :, :10] *= scaling_factor
    
    return hidden_states


def extract_answer(output_string, task_type="yes_no"):
    """
    Extracts the answer from the output string based on the task type.

    Parameters:
    output_string (str): The output string.
    task_type (str): The type of task. Must be either "yes_no" or "multiple_choice".

    Returns:
    int: 
        1 if "yes" or "A" 
        0 if "no" or "B"
        -1 if no relevant answer is found.
        Raises a ValueError if an unsupported task_type is provided.
    """

    def find_word_position(string, word):
        pattern = r'\b' + re.escape(word) + r'\b'
        match = re.search(pattern, string, re.IGNORECASE)
        if match:
            return match.start()
        return -1
    
    if task_type not in ["yes_no", "multiple_choice"]:
        raise ValueError("Task type not supported. Must be 'yes_no' or 'multiple_choice'.")
    
    if task_type == "yes_no":
        position_yes_and_a = find_word_position(output_string, "yes")
        position_no_and_b = find_word_position(output_string, "no")
    elif task_type == "multiple_choice":
        position_yes_and_a = find_word_position(output_string, "A")
        position_no_and_b = find_word_position(output_string, "B")

    if position_yes_and_a == -1 and position_no_and_b == -1:
        print(f"No answer found in the output string: {output_string}.")
        return -1
    elif position_yes_and_a != -1 and position_no_and_b != -1:
        return 1 if position_yes_and_a < position_no_and_b else 0
    else:
        return 0 if position_yes_and_a == -1 else 1


def get_scores(scores):
    """
    Calculate various scores based on the given results.

    Args:
        scores (dict or list): A dictionary or list containing results where each result can be:
            - dict: {id: {"q0_i0": 1 or 0, "q0_i1": 1 or 0, "q1_i0": 1 or 0, "q1_i1": 1 or 0}, ...}
            - list: [[q0_i0 (1 or 0), q0_i1 (1 or 0), q1_i0 (1 or 0), q1_i1 (1 or 0)], ...]

    The keys "q0_i0", "q0_i1", "q1_i0", "q1_i1" represent combinations of questions and images:
        - "q0_i0" means question_0 on image_0 
        - "q0_i1" means question_0 on image_1 
        - "q1_i0" means question_1 on image_0 
        - "q1_i1" means question_1 on image_1 

    Returns:
        dict: A dictionary containing the calculated scores:
            - 'Q_Acc': Average question score
            - 'I_Acc': Average image score
            - 'Acc': Average binary VQA score
            - 'G_Acc': Average group score
    """
    Q_Acc = 0.0
    I_Acc = 0.0
    Acc = 0.0
    G_Acc = 0.0

    num_samples = len(scores)

    def calculate_image_score(result):
        image_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q1_i0"] == 0.0:
                image_correct += 1
            if result["q1_i1"] == 1.0 and result["q0_i1"] == 0.0:
                image_correct += 1
        elif isinstance(result, list):
            if result[0] == 1.0 and result[2] == 0.0:
                image_correct += 1
            if result[3] == 1.0 and result[1] == 0.0:
                image_correct += 1
        return image_correct
    
    def calculate_question_score(result):
        text_correct = 0
        if isinstance(result, dict):
            if result["q0_i0"] == 1.0 and result["q0_i1"] == 0.0:
                text_correct += 1
            if result["q1_i1"] == 1.0 and result["q1_i0"] == 0.0:
                text_correct += 1
        else:
            if result[0] == 1.0 and result[1] == 0.0:
                text_correct += 1
            if result[3] == 1.0 and result[2] == 0.0:
                text_correct += 1
        return text_correct

    def calculate_binary_score(result):
        binary_score_correct = 0
        if isinstance(result, dict):
            binary_score_correct += 1 if result["q0_i0"] == 1.0 else 0
            binary_score_correct += 1 if result["q0_i1"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i0"] == 0.0 else 0
            binary_score_correct += 1 if result["q1_i1"] == 1.0 else 0
        else:
            binary_score_correct += 1 if result[0] == 1.0 else 0
            binary_score_correct += 1 if result[1] == 0.0 else 0
            binary_score_correct += 1 if result[2] == 0.0 else 0
            binary_score_correct += 1 if result[3] == 1.0 else 0

        return binary_score_correct

    def calculate_group(result):
        group_correct = 0
        if calculate_question_score(result) == 2 and calculate_image_score(result) == 2:
            group_correct += 1
        
        return group_correct
    
    if isinstance(scores, dict):
        for _, result in scores.items():
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)
    else:
        for result in scores:
            Q_Acc += calculate_question_score(result)
            I_Acc += calculate_image_score(result)
            Acc += calculate_binary_score(result)
            G_Acc += calculate_group(result)

    results = {
        'Q_Acc': Q_Acc / float(num_samples * 2),
        'I_Acc': I_Acc / float(num_samples * 2),
        'Acc': Acc / float(num_samples * 4),
        'G_Acc': G_Acc / num_samples
    }

    return results


class AdjustedQwen2_5_VL(Qwen2_5_VLForConditionalGeneration):
    r"""Qwen2-5-VL with per-step embedding adjust hook."""
    def __init__(self, config, *, adjust_function=None, **kw):
        super().__init__(config, **kw)
        self.adjust_function = adjust_function

    # ← Qwen 이 이미 오버라이드한 시그니처를 1:1 로 복사
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        
        # ──────────────────────────────────────────────────────────────────
        # 1) 먼저 Qwen 원본 prepare 로직 실행
        # ──────────────────────────────────────────────────────────────────
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )

        # ──────────────────────────────────────────────────────────────────
        # 2) *매 스텝* 마지막 토큰 임베딩 교체
        # ──────────────────────────────────────────────────────────────────
        if self.adjust_function is not None and past_key_values is not None:
            ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
            ids     = model_inputs.get(ids_key)

            if ids is not None:                       # (B, S)  — S ≥ 1
                # 1) 전체 시퀀스 임베딩
                embeds = self.get_input_embeddings()(ids)          # (B, S, H)

                # 2) 마지막 토큰만 변형
                last_emb = embeds[:, -1:, :]                       # (B,1,H)
                embeds[:, -1:, :] = self.adjust_function(last_emb)

                # 3) IDs 대신 변형된 전체 임베딩 사용
                model_inputs[ids_key]         = None
                model_inputs["inputs_embeds"] = embeds


        return model_inputs


def advanced_hidden_manipulation_fn(hidden_states, step_idx=None):
    """
    고급 hidden state 조작 함수 예시
    
    Args:
        hidden_states (torch.Tensor): 조작할 hidden states
        step_idx (int, optional): 현재 생성 단계
        
    Returns:
        torch.Tensor: 조작된 hidden states
    """
    # 원본 히든 스테이트 복사
    modified_hidden = hidden_states.clone()
    
    # 특정 시나리오에 따라 다른 조작 적용
    if step_idx is not None:
        # 1. 토큰 생성 초기 단계에서는 특정 방향으로 약간 편향
        if step_idx < 5:
            # 특정 임베딩 방향으로 미세 조정 (예: 감정, 스타일 등의 방향)
            direction_vector = torch.randn_like(hidden_states[:, 0, :])
            direction_vector = direction_vector / direction_vector.norm()
            
            # 마지막 위치의 임베딩만 조작
            scale = 0.1 * (5 - step_idx) / 5  # 초기에 더 강하게, 점점 약해짐
            modified_hidden[:, -1, :] += scale * direction_vector
            
        # 2. 중간 단계에서는 다른 조작 적용
        elif 5 <= step_idx < 15:
            # 예: 특정 차원 강화
            dimension_mask = torch.zeros_like(modified_hidden[:, -1, :])
            dimension_mask[:, 100:200] = 1.0  # 특정 차원(예: 100-200)만 강화
            
            # 특정 차원을 강화하는 효과 적용
            modified_hidden[:, -1, :] *= (1.0 + 0.05 * dimension_mask)
            
        # 3. 후반 단계에서는 다른 조작 적용
        else:
            # 예: 특정 응답으로 가이드 (예: 긍정적인 응답)
            if hasattr(hidden_states, 'device'):
                positive_direction = torch.tensor([0.1, 0.2, -0.1], device=hidden_states.device)
                positive_direction = torch.nn.functional.pad(
                    positive_direction, 
                    (0, hidden_states.size(-1) - 3)
                )
                
                # 방향 정규화
                positive_direction = positive_direction / positive_direction.norm()
                
                # 마지막 위치의 임베딩을 긍정적인 방향으로 약간 조정
                modified_hidden[:, -1, :] += 0.05 * positive_direction
    
    return modified_hidden


if __name__ == "__main__":
    # 1. Load dataset from HuggingFace
    dataset = load_dataset("BaiqiL/NaturalBench")

    # 2. Use NaturalBench: construct 1900*4 [question, image, correct_answer, question_type] samples
    naturalbench = []
    for item in dataset["train"]:
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_0"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_0"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_0"], item["Image_0_Question_1"], item['Question_Type']])
        naturalbench.append([item["Question_1"] + SUFFIX_FOR_VQA[item["Question_Type"]], item["Image_1"], item["Image_1_Question_1"], item['Question_Type']])
    
    # sample_count = len(naturalbench)  # 전체 데이터셋 사용
    sample_count = 10 # 적은 수의 샘플로 테스트하려면 이 줄 주석 해제
    
    # 3. Test Models with custom generation capabilities
    ## 3.1 Load the model as our custom model class
    model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
    model = AdjustedQwen2_5_VL.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name, min_pixels=256*28*28, max_pixels=1280*28*28)
    
    # 3.1 Run the model on NaturalBench with original generation
    original_output_file_path = "./Qwen_2_5_VL_3B_original_generation.json"
    model.adjust_function = None
    if os.path.exists(original_output_file_path):
        with open(original_output_file_path, 'r', encoding='utf-8') as file:
            original_output_file = json.load(file)
    else:
        original_output_file = []
        for i in tqdm(range(sample_count)):
            item = naturalbench[i]
            question = item[0]
            image = item[1]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :].cpu() for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            original_output_file.append(output_text[0] if isinstance(output_text, list) else output_text)

            del inputs, generated_ids
            torch.cuda.empty_cache()

 
        with open(original_output_file_path, 'w') as json_file:
            json.dump(original_output_file, json_file, indent=4)
    
    ## 3.2 Run the model on NaturalBench with custom generation
    adjusted_output_file_path = "./Qwen_2_5_VL_3B_custom_generation.json"
    model.adjust_function = advanced_hidden_manipulation_fn

    if os.path.exists(adjusted_output_file_path):
        with open(adjusted_output_file_path, 'r', encoding='utf-8') as file:
            adjusted_output_file = json.load(file)
    else:
        adjusted_output_file = []
        for i in tqdm(range(sample_count)):
            item = naturalbench[i]
            question = item[0]
            image = item[1]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": question},
                    ],
                }
            ]

            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)

            generated_ids_trimmed = [
                out_ids[len(in_ids) :].cpu() for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            adjusted_output_file.append(output_text[0] if isinstance(output_text, list) else output_text)

            del inputs, generated_ids
            torch.cuda.empty_cache()

 
        with open(adjusted_output_file_path, 'w') as json_file:
            json.dump(adjusted_output_file, json_file, indent=4)

    # 4. Extract the answer from outputs
    assert len(adjusted_output_file) == sample_count  # 실제 처리된 샘플 수 검증
    answers = {}
    number_answered_samples = len(adjusted_output_file) // 4  # 그룹당 4개 샘플
    
    for i in range(number_answered_samples):
        answers[i] = {
            "q0_i0": extract_answer(adjusted_output_file[i*4], naturalbench[i*4][3]),
            "q0_i1": extract_answer(adjusted_output_file[i*4+1], naturalbench[i*4+1][3]),
            "q1_i0": extract_answer(adjusted_output_file[i*4+2], naturalbench[i*4+2][3]),
            "q1_i1": extract_answer(adjusted_output_file[i*4+3], naturalbench[i*4+3][3])
        }

    # 5. Calculate the scores
    adjusted_scores = get_scores(answers)
    
    # 6. 원본 결과와 커스텀 생성 결과 비교 (선택적)
    if os.path.exists(original_output_file_path):
        with open(original_output_file_path, 'r', encoding='utf-8') as file:
            original_output = json.load(file)
            
        original_answers = {}
        for i in range(number_answered_samples):
            original_answers[i] = {
                "q0_i0": extract_answer(original_output[i*4], naturalbench[i*4][3]),
                "q0_i1": extract_answer(original_output[i*4+1], naturalbench[i*4+1][3]),
                "q1_i0": extract_answer(original_output[i*4+2], naturalbench[i*4+2][3]),
                "q1_i1": extract_answer(original_output[i*4+3], naturalbench[i*4+3][3])
            }
            
        original_scores = get_scores(original_answers)
        print("Original scores:", original_scores)
        print("Custom generation scores:", adjusted_scores)
        
        # 점수 차이 계산
        score_diff = {k: adjusted_scores[k] - original_scores[k] for k in adjusted_scores.keys()}
        print("Score differences (Custom - Original):", score_diff)