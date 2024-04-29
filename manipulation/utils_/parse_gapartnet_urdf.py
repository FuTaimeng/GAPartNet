
import xml.etree.ElementTree as ET
import json, glob, tqdm

def parse_gapartnet_urdf(urdf_path):
    tree_urdf = ET.parse(urdf_path)
    root_urdf = tree_urdf.getroot()
    parent_child_pairs = []
    for joint in root_urdf.iter('joint'):
        joint_name = joint.attrib['name']
        joint_type = joint.attrib['type']
        assert len(list(joint.iter('parent'))) == 1
        assert len(list(joint.iter('child'))) == 1
        parent_child_pairs.append((joint.find('parent').attrib['link'], joint.find('child').attrib['link'], joint_name))
    return parent_child_pairs

def add_parent_child_info_to_anno(id):
    anno_path = f"/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/{id}/link_annotation_gapartnet.json"
    urdf_path = f"/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/{id}/mobility_annotation_gapartnet.urdf"
    parent_child_pairs = parse_gapartnet_urdf(urdf_path)
    anno = json.load(open(anno_path, 'r'))
    for anno_part in anno:
        part_id = anno_part['link_name']
        if 'children' not in anno_part:
            anno_part['children'] = []
        if 'parents' not in anno_part:
            anno_part['parents'] = []
        if "children_joint" not in anno_part:
            anno_part["children_joint"] = []
        if "parents_joint" not in anno_part:
            anno_part["parents_joint"] = []
        for parent, child, joint_name in parent_child_pairs:
            # import pdb; pdb.set_trace()
            if part_id == parent:
                anno_part['children'].append(child)
                anno_part['children_joint'].append(joint_name)
            if part_id == child:
                anno_part['parents'].append(parent)
                anno_part['parents_joint'].append(joint_name)
                
    # save the updated annotation
    anno_path_new = f"/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/{id}/link_annotation_gapartnet_with_parent_child.json"
    with open(anno_path_new, 'w') as f:
        json.dump(anno, f, indent=4)
    # import pdb; pdb.set_trace()

for path in tqdm.tqdm(glob.glob(f"/home/haoran/Projects/Part/GAPartNet/manipulation/assets/gapartnet/*/mobility_annotation_gapartnet.urdf")):
    id = path.split('/')[-2]
    print(id)
    add_parent_child_info_to_anno(id)
