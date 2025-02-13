import ast
import json
from collections import defaultdict
from nltk.corpus import wordnet as wn

def load_imagenet_data(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        data = file.read()
    return ast.literal_eval(data)

def map_to_synsets(imagenet_data):
    """
    Synset ID를 WordNet Synset 객체로 매핑.
    매핑되지 않은 ID를 로깅.
    """
    id_to_label = {}
    id_to_synset = {}
    unmapped_ids = []  # 매핑되지 않은 ID 기록

    for key, item in imagenet_data.items():
        syn_id = item["id"]  # "one.s.01" 형식
        label_str = item["label"]

        try:
            # WordNet Synset 객체로 변환
            synset_obj = wn.synset(syn_id)
            id_to_synset[key] = synset_obj
            id_to_label[key] = label_str
        except Exception as e:
            unmapped_ids.append((syn_id, str(e)))

    if unmapped_ids:
        print(f"Unmapped Synset IDs: {unmapped_ids}")

    return id_to_label, id_to_synset

def group_synsets_bottom_up(id_to_synset, min_size=3):
    # Synset-leaf 매핑 사전 생성
    synset_to_leaves = defaultdict(set)
    for syn_id, syn in id_to_synset.items():
        current_syn = syn
        visited = set()
        while current_syn not in visited:
            visited.add(current_syn)
            synset_to_leaves[current_syn].add(syn_id)
            parents = current_syn.hypernyms()
            if not parents:
                break
            current_syn = parents[0]

    # Synset 깊이 계산
    depth_cache = {}
    def get_depth(syn):
        if syn in depth_cache:
            return depth_cache[syn]
        parents = syn.hypernyms()
        depth = 0 if not parents else 1 + max(get_depth(p) for p in parents)
        depth_cache[syn] = depth
        return depth

    # 깊이 기준 정렬 (깊은 synset 먼저 처리)
    all_synsets = sorted(synset_to_leaves.keys(), key=lambda x: -get_depth(x))

    assigned = set()
    groups = []
    group_names = []
    processed_roots = set()  # 중복 루트 처리 방지

    # 1단계: 일반 그룹화
    for syn in all_synsets:
        leaves = synset_to_leaves[syn]
        unassigned = leaves - assigned
        if len(unassigned) >= min_size:
            groups.append(unassigned)
            group_names.append(syn.lemmas()[0].name() if syn.lemmas() else syn.name())
            assigned.update(unassigned)

    # 2단계: 남은 클래스 처리 (루트 노드 강제 그룹화)
    remaining = set(id_to_synset.keys()) - assigned
    for syn_id in remaining:
        current_syn = id_to_synset[syn_id]
        
        # 루트 노드까지 상승
        while True:
            parents = current_syn.hypernyms()
            if not parents:
                break
            current_syn = parents[0]

        # 이미 처리된 루트 건너뛰기
        if current_syn in processed_roots:
            continue
            
        # 해당 루트의 모든 잔여 클래스 수집
        root_leaves = synset_to_leaves.get(current_syn, set())
        unassigned_in_root = root_leaves & remaining
        
        # 강제 그룹화 (크기 무관)
        if unassigned_in_root:
            groups.append(unassigned_in_root)
            group_name = current_syn.lemmas()[0].name() if current_syn.lemmas() else "root_group"
            group_names.append(group_name)
            assigned.update(unassigned_in_root)
            processed_roots.add(current_syn)

    return groups, group_names, 0

def split_multiword_labels(groups, id_to_label):
    processed_groups = []
    for group in groups:
        processed_group = []
        for syn_id in group:
            label = id_to_label.get(syn_id, "")
            labels = [item.strip() for item in label.split(",")]
            processed_group.extend(labels)
        processed_groups.append(processed_group)
    return processed_groups

def calculate_group_stats(groups):
    group_sizes = [len(group) for group in groups]
    avg_size = sum(group_sizes) / len(group_sizes) if group_sizes else 0
    min_size = min(group_sizes) if group_sizes else 0
    max_size = max(group_sizes) if group_sizes else 0
    return avg_size, min_size, max_size

def save_groups_to_file(processed_groups, group_names, filepath):
    if len(processed_groups) != len(group_names):
        print(f"Error: Group names count {len(group_names)} vs groups count {len(processed_groups)}")

    group_dict = {}
    for name, group in zip(group_names, processed_groups):
        if name in group_dict:
            group_dict[name].extend(group)
        else:
            group_dict[name] = group

    with open(filepath, "w", encoding="utf-8") as file:
        json.dump(group_dict, file, indent=4, ensure_ascii=False)
    print(f"Groups saved to {filepath}")

if __name__ == "__main__":
    filepath = "./imagenet_label_to_wordnet_synset.txt"
    output_filepath = "./grouped_imagenet_classes.json"

    imagenet_data = load_imagenet_data(filepath)
    id_to_label, id_to_synset = map_to_synsets(imagenet_data)

    groups, group_names, _ = group_synsets_bottom_up(id_to_synset, min_size=3)

    avg_size, min_size, max_size = calculate_group_stats(groups)
    print(f"Total groups: {len(groups)}")
    print(f"Avg size: {avg_size:.2f}, Min: {min_size}, Max: {max_size}")

    processed_groups = split_multiword_labels(groups, id_to_label)
    save_groups_to_file(processed_groups, group_names, output_filepath)