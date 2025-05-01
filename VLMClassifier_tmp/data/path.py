import json

file_path = "/data1/bubble3jh/coe/VLMClassifier/data/imagenet.jsonl"
old_path = "/pasteur/u/yuhuiz/data/ImageNet"
new_path = "/data1/bubble3jh/data"

# 읽고 수정
with open(file_path, "r") as f:
    lines = f.readlines()

# 수정 내용 적용
with open(file_path, "w") as f:
    for line in lines:
        item = json.loads(line)
        if "image" in item and old_path in item["image"]:
            item["image"] = item["image"].replace(old_path, new_path)
        f.write(json.dumps(item) + "\n")
