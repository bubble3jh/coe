import inspect
import types
import functools
import torch
import transformers
from transformers.generation.utils import GenerationMixin
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ----------------------------------------------------------------------
# 0. 헬퍼: bound-method / partial → 실제 함수 객체 추출
# ----------------------------------------------------------------------
def unwrap_func(f):
    """
    인스턴스에서 꺼낸 bound-method나 functools.partial 등을
    순수 function 객체로 풀어서 반환.
    """
    while isinstance(f, (types.MethodType, functools.partial)):
        f = f.__func__ if isinstance(f, types.MethodType) else f.func
    return f


# ----------------------------------------------------------------------
# 1. 모델 로드
# ----------------------------------------------------------------------
model_name = "Qwen/Qwen2.5-VL-3B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
# (토크나이저는 여기선 필수는 아니지만, 이후 실험용으로 함께 로드)


# ----------------------------------------------------------------------
# 2. prepare_inputs_for_generation 구현 비교
# ----------------------------------------------------------------------
qwen_prepare_func = unwrap_func(model.prepare_inputs_for_generation)
base_prepare_func = unwrap_func(GenerationMixin.prepare_inputs_for_generation)

uses_base_mixin = qwen_prepare_func.__code__ is base_prepare_func.__code__

if uses_base_mixin:
    print("Qwen 모델은 *GenerationMixin*의 기본 prepare_inputs_for_generation을 사용합니다.")
    source_to_copy = inspect.getsource(base_prepare_func)
else:
    print("Qwen 모델은 **자체적으로** prepare_inputs_for_generation을 오버라이드합니다.")
    source_to_copy = inspect.getsource(qwen_prepare_func)

print("\n--- 복사할 소스 코드 ---\n")
print(source_to_copy)
