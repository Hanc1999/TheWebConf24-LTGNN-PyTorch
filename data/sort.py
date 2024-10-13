# read tri_graph_uidx2pidx_valid.json and sort them according to the key

import json
import os

def json_sort(json_file_path, txt_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)

        # sort the data by converting keys to integers
        data = dict(sorted(data.items(), key=lambda item: int(item[0])))

    with open(txt_file_path, 'w') as txt_file:
        for key, values in data.items():
            line = f"{key} " + " ".join(map(str, values)) + "\n"
            txt_file.write(line)

json_file_path = './mba/tri_graph_tidx2pidx.json'
txt_file_path = './mba/tidx2pidx.txt'

json_sort(json_file_path, txt_file_path)