import glob
import os
import json
task_name = "open drawer with handle"
paths = glob.glob('/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/*/link_annotation_gapartnet_with_parent_child.json')
tasks = []
for path in paths:
    id = path.split('/')[-2]
    anno = json.load(open(path, 'r'))
    for part_anno in anno:
        # check this part is drawer
        if not (part_anno["is_gapart"] and part_anno['category'] == 'slider_drawer'):
            continue
        # check the next part is handle
        for child in part_anno['children']:
            child_anno = next(filter(lambda x: x['link_name'] == child, anno))
            if child_anno["is_gapart"] and child_anno['category'][-6:] == 'handle':
                # import pdb; pdb.set_trace()
                task_dict = {
                    "id": id,
                    "part_anno": part_anno,
                    "child_anno": child_anno,
                    "task_name": task_name,
                }
                tasks.append(task_dict)
                print(part_anno)
                # break
print(len(tasks))
task_name_str = task_name.replace(" ", "_")
json.dump(tasks, open(f'/home/haoran/Projects/Part/GAPartNet/manipulation/assets/tasks_{task_name_str}.json', 'w'), indent=4)