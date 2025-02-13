import ast
import json
from nltk.corpus import wordnet as wn

# Step 1: 텍스트 파일 읽어서 데이터 로드
def load_imagenet_data(filepath):
    """
    ./imagenet_label_to_wordnet_synset.txt 파일에서 데이터를 로드하여
    Synset ID와 클래스 이름을 매핑한 딕셔너리를 반환
    """
    with open(filepath, "r", encoding="utf-8") as file:
        data = file.read()
    return ast.literal_eval(data)

# Step 2: Synset ID를 WordNet Synset 객체로 매핑
def map_to_synsets(imagenet_data):
    """
    Synset ID를 WordNet Synset 객체로 매핑
    """
    id_to_label = {}
    id_to_synset = {}

    for key, item in imagenet_data.items():
        syn_id = item['id']
        label_str = item['label']

        offset_str, pos_letter = syn_id.split("-")
        offset = int(offset_str)
        pos = pos_letter

        try:
            synset_obj = wn.synset_from_pos_and_offset(pos, offset)
            id_to_synset[key] = synset_obj
            id_to_label[key] = label_str
        except:
            continue

    return id_to_label, id_to_synset

# Bottom-Up 방식으로 그룹화 수행
def group_synsets_bottom_up(id_to_synset, min_size=3, max_size=20):
    """
    WordNet Synset 객체를 기반으로 Bottom-Up 방식으로 그룹화 수행.
    하위 그룹의 모든 하위 Synset까지 탐색하여, 분리 가능한 가장 낮은 레벨부터 분리.
    """
    unassigned = set(id_to_synset.keys())  # 아직 그룹에 할당되지 않은 Synset ID
    groups = []
    group_names = []
    oversized_count = 0

    def gather_hyponyms(syn):
        """
        특정 Synset을 포함한 모든 하위(hyponym) Synset ID를 반환
        """
        hyponyms = set()
        for descendant in syn.closure(lambda s: s.hyponyms()):
            for key, val in id_to_synset.items():
                if val == descendant and key in unassigned:
                    hyponyms.add(key)
        return hyponyms

    def can_divide_group(syn):
        """
        주어진 Synset의 하위 Synset들이 모두 `min_size` 이상인지 확인하여,
        가장 낮은 레벨에서 분리 가능한지 검사.
        """
        children = syn.hyponyms()
        divisible_groups = []
        remaining_count = 0

        # 각 하위 Synset의 하위 그룹 크기를 계산
        for child in children:
            child_hyponyms = gather_hyponyms(child)
            if len(child_hyponyms) >= min_size:
                divisible_groups.append(child_hyponyms)
            else:
                remaining_count += len(child_hyponyms)

        # 남은 그룹이 min_size 이상인지 확인
        if remaining_count >= min_size:
            return divisible_groups, remaining_count
        else:
            return None, 0

    while unassigned:
        syn_id = unassigned.pop()
        syn = id_to_synset[syn_id]
        current_syn = syn

        while True:
            children = gather_hyponyms(current_syn)

            # 하위 그룹 확인 및 분리 가능한지 검사
            divisible_groups, remaining_count = can_divide_group(current_syn)

            if divisible_groups:
                # 하위 그룹을 분리 가능한 경우
                for group in divisible_groups:
                    groups.append(group)
                    unassigned -= group

                    # 그룹 이름 설정
                    group_names.append(current_syn.lemmas()[0].name())

                # 남은 그룹 처리
                if remaining_count >= min_size:
                    remaining_group = gather_hyponyms(current_syn) - set.union(*divisible_groups)
                    groups.append(remaining_group)
                    unassigned -= remaining_group
                    group_names.append(current_syn.lemmas()[0].name())
                break

            # 현재 Synset이 그룹화 가능한 경우
            if len(children) >= min_size:
                groups.append(children)
                unassigned -= children
                group_names.append(current_syn.lemmas()[0].name())
                break

            # 상위 Synset으로 이동
            parents = current_syn.hypernyms()
            if not parents:
                # 최상위 Synset에 도달한 경우
                groups.append(children)
                unassigned -= children
                group_names.append(current_syn.lemmas()[0].name() if current_syn.lemmas() else "Unknown")
                break

            current_syn = parents[0]  # 첫 번째 상위 Synset으로 이동

    return groups, group_names, oversized_count


# Step 4: 후처리 - 클래스 이름 분리
def split_multiword_labels(groups, id_to_label):
    """
    그룹화된 클래스 이름 중 다중 단어 표현(e.g., "pencil box, pencil case")을
    개별 항목으로 분리
    """
    processed_groups = []
    for group in groups:
        processed_group = []
        for syn_id in group:
            label = id_to_label[syn_id]
            labels = [item.strip() for item in label.split(",")]
            processed_group.extend(labels)
        processed_groups.append(processed_group)
    return processed_groups

# Step 5: 그룹 크기 통계 계산
def calculate_group_stats(groups):
    """
    그룹 크기의 평균, 최소, 최대를 계산
    """
    group_sizes = [len(group) for group in groups]
    avg_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
    min_size = min(group_sizes) if group_sizes else 0
    max_size = max(group_sizes) if group_sizes else 0
    return avg_size, min_size, max_size

# Step 6: 그룹 데이터를 로컬에 저장
def save_groups_to_file(processed_groups, group_names, filepath):
    """
    후처리된 그룹 데이터를 dict 형식으로 변환한 후 JSON 파일로 저장
    """
    group_dict = {
        group_name: group
        for group_name, group in zip(group_names, processed_groups)
    }
    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(group_dict, file, indent=4, ensure_ascii=False)
    print(f"그룹 데이터가 {filepath}에 저장되었습니다.")

# Main: 데이터 로드 및 처리
if __name__ == "__main__":
    filepath = "./imagenet_label_to_wordnet_synset.txt"
    output_filepath = "./grouped_imagenet_classes.json"

    # 데이터 로드
    imagenet_data = load_imagenet_data(filepath)

    # Synset 매핑
    id_to_label, id_to_synset = map_to_synsets(imagenet_data)

    # 그룹화 수행
    groups, group_names, oversized_count = group_synsets_bottom_up(
        id_to_synset,
        min_size=3,
        max_size=20
    )

    # 그룹 크기 통계 출력
    avg_size, min_size, max_size = calculate_group_stats(groups)
    print(f"그룹 개수: {len(groups)}")
    print(f"그룹 평균 크기: {avg_size:.2f}, 최소 크기: {min_size}, 최대 크기: {max_size}")
    print(f"크기가 {max_size}을 초과한 그룹 개수: {oversized_count}")

    # 후처리 - 다중 클래스 이름 분리
    processed_groups = split_multiword_labels(groups, id_to_label)

    # 결과 출력
    for idx, group in enumerate(processed_groups[:5]):
        print(f"[{idx}] {group_names[idx]} (size={len(group)}): {group}")

    # JSON 저장
    save_groups_to_file(processed_groups, group_names, output_filepath)
