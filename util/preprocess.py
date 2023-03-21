def construct_data(data, feature_map, labels=0):
    res = []
    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, "not exist in data")

    sample_n = len(res[0])
    if type(labels) == int:
        res.append([labels] * sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def build_loc_net(struc, all_features, feature_map=[]):
    index_feature_map = feature_map
    edge_indexes = [[], []]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f"Error: {child} not in index_feature_map")

            c_index = index_feature_map.index(child)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes
