import copy

def modify_node_types(node_configurations: dict, nodetype: str, key: str, value):
    node_configurations = copy.deepcopy(node_configurations)
    
    for node_config in node_configurations:
        if node_config['nodetype'] == nodetype:
            node_config[key] = value

    return node_configurations
