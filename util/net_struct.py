def get_feature_map(dataset):
    feature_file = open(f"./data/{dataset}/list.txt", "r")
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list


def get_fc_graph_struc(dataset):
    feature_file = open(f"./data/{dataset}/list.txt", "r")
    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map
