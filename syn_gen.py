import json
from nltk.corpus import wordnet
from imagenet_classnames import openai_classnames  # 이미지넷 클래스 이름 가져오기
from tqdm import tqdm  # 진행 상황 표시를 위한 tqdm 추가

# 결과를 저장할 딕셔너리
imagenet_label_to_wordnet_synset = {}
missing_count = 0  # 매핑 실패한 개수를 기록
unmatched_labels = []  # 매칭되지 않은 레이블 저장

# 각 label에 대해 synset 정보를 매핑
for idx, label in tqdm(enumerate(openai_classnames), total=len(openai_classnames), desc="Mapping labels"):
    # WordNet에서 synset 검색
    words = [word.strip() for word in label.split(",")]  # label에서 모든 이름 추출
    synset = None

    # 모든 단어에 대해 정의 포함 여부를 체크
    for word in words:
        synsets = wordnet.synsets(word)
        if synsets:
            for candidate in synsets:
                if any(w in candidate.definition() or w in candidate.lemma_names() for w in words):
                    synset = candidate
                    break
            if synset:
                break

    # 단어 포함 관계로 재검색
    if not synset:
        for syn in wordnet.all_synsets():
            if any(word in syn.name() or word in syn.definition() or word in syn.lemma_names() for word in words):
                synset = syn
                break

    # 추가: 쉼표 포함 전체 레이블을 정의에서 검색
    if not synset:
        for syn in wordnet.all_synsets():
            if any(label_part.strip() in syn.definition() for label_part in label.split(",")):
                synset = syn
                break

    # 추가: Synset의 lemma 이름이 레이블에 포함되는지 확인
    if not synset:
        for syn in wordnet.all_synsets():
            if any(lemma in label for lemma in syn.lemma_names()):
                synset = syn
                break

    if synset:
        synset_info = {
            "id": synset.name(),
            "label": label,
            "uri": f"http://wordnet-rdf.princeton.edu/wn30/{synset.name()}"
        }
    else:
        print(f'{label} is not matched')
        synset_info = {
            "id": None,
            "label": label,
            "uri": None
        }
        missing_count += 1
        unmatched_labels.append(label)

    imagenet_label_to_wordnet_synset[idx] = synset_info

# 결과를 파일에 저장
output_file = "imagenet_label_to_wordnet_synset.txt"
with open(output_file, "w") as f:
    json.dump(imagenet_label_to_wordnet_synset, f, indent=2)

# 매칭되지 않은 레이블을 별도 저장
unmatched_file = "unmatched_labels.txt"
with open(unmatched_file, "w") as f:
    for label in unmatched_labels:
        f.write(label + "\n")

print(f"Mapping saved to {output_file}")
print(f"Number of labels without a matching synset: {missing_count}")
print(f"Unmatched labels saved to {unmatched_file}")
