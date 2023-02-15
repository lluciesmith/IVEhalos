import numpy as np


class Params(object):
    def __init__(self, input_file_name):
        """
        Creates attributes for every row element of the `input_file_name`.
        :param input_file_name: rows in format `attribute (type): label`
        """
        dtypes = {'str': str, 'float': float, 'int': int, 'tuple': tuple, 'list': list}

        with open(input_file_name, 'r') as input_file:
            names_list = [line.strip() for line in input_file if line.strip() and not line.startswith('#')]

        for row in names_list:
            attr, label = get_attribute_label_from_row(row, dtypes)
            self.__dict__[attr] = label


class Architecture:
    def __init__(self, input_file):
        """
        Creates dictionaries with model architecture parameters.
        :param input_file_name: rows in format `attribute (type): label`
        """
        self.dtypes = {'str': str, 'float': float, 'int': int, 'tuple': tuple, 'list': list}

        with open(input_file, "r") as input_file:
            names_list = [line.strip() for line in input_file if line.strip() and not line.startswith('#')]

        convs = [name.strip("[]") for name in names_list if "conv" in name]
        param_conv = {key: {} for key in convs}
        self.get_dictionary_each_layer(param_conv, convs, names_list)

        dense = [name.strip("[]") for name in names_list if "dense" in name]
        dense.append("last")
        param_fcc = {key: {} for key in dense}
        self.get_dictionary_each_layer(param_fcc, dense, names_list)

        self.param_conv = param_conv
        self.param_fcc = param_fcc

    def get_dictionary_each_layer(self, result, layer_type, all_layer_names):
        _layers_names = np.array(all_layer_names)[["[" in l for l in all_layer_names]]

        for i, layer in enumerate(layer_type):
            layer_idx = all_layer_names.index("[" + layer_type[i] + "]")
            if layer_idx == len(all_layer_names) - 1 or layer == "last":
                layer_params = all_layer_names[layer_idx + 1:]
            else:
                next_layer_name = _layers_names[np.where(_layers_names == all_layer_names[layer_idx])[0] + 1][0]
                next_layer_idx = all_layer_names.index(next_layer_name)
                layer_params = all_layer_names[layer_idx + 1:next_layer_idx]

            for row in layer_params:
                attr, label = get_attribute_label_from_row(row, self.dtypes)
                result[layer][attr] = label

        return result


def get_attribute_label_from_row(row, dtypes):
    attr = row[: row.find('(')].strip()
    label = row[row.find(':') + 1:].strip(' "').replace("'", "")
    type_attr = row[row.find('(') + 1: row.find(')')].strip()

    if type_attr.find(",") != -1:
        # this takes into account the fact that `type` can be a tuple, e.g. (list, float)
        type0, type1 = [elem.strip('[( "])') for elem in type_attr.split(",")]
        ls = [elem.strip('[(" ])') for elem in label.split(",")]

        # transform each element of `label` into data type `type1`
        ls_dtype1 = [True if x == "True" else False if x == "False" else None if x == "None" else dtypes[type1](x) for x in ls]

        # transform `label` into data type `type2`
        label = dtypes[type0](ls_dtype1)

    else:
        # transform `label` into data type `type_attr`
        label = True if label == "True" else False if label == "False" else None if label == "None" else dtypes[type_attr](label)

    if "thresholds" in attr:
        label = np.array(label)
    return attr, label
