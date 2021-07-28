import xml.etree.ElementTree as ET


def append_xml(in_path, out_path, append_dict):
    tree = ET.parse(in_path)
    root = tree.getroot()
    for name in append_dict.keys():
        n = ET.Element(name)
        if isinstance(append_dict[name] , dict):
            for prop in append_dict[name].keys():
                n.set(prop, append_dict[name][prop])
        else:
            n.text = str(append_dict[name])
        root.append(n)
    tree.write(out_path)