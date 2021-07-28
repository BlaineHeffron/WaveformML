import xml.etree.ElementTree as ET
import sys

from xml.dom import minidom
from utils.util import get_run_info, get_file_md5, check_path
from os.path import expanduser


def append_xml(in_path, out_path, append_dict, parent=None):
    if parent is None:
        tree = ET.parse(in_path)
        root = tree.getroot()
    else:
        root = parent
    for name in append_dict.keys():
        if name == "_PROP_":
            #special case, describes properties of current node
            for key in append_dict[name]:
                root.set(key, append_dict[name][key])
            continue
        n = ET.Element(name)
        if isinstance(append_dict[name] , dict):
            append_xml(in_path, out_path, append_dict[name], n)
        else:
            n.text = append_dict[name]
        root.append(n)
    if parent is None:
        tree.write(out_path)

def main():
    dir = "~/projects/neutrino_ML/data/waveforms/type_rn"
    inpath = check_path(dir + "/s015_f00011_ts1520333492_WFNorm.h5.xml")
    outpath = expanduser(dir + "/s015_f00011_ts1520333492_WFNormModified.h5.xml")
    input_dict = {"WaveformML": {
        "AnalysisStep": {
            "_PROP_": {
                "code": "test"
            },
            "input": {
                "_PROP_": {
                    "file": inpath,
                    "md5": get_file_md5(inpath)
                }
            },
            "output": {
                "_PROP_": {
                    "file": outpath[0:-4]
                }
            },
            "test_step":{
                "_PROP_": {
                    "setting1": "blah",
                    "setting2": "yeah"
                }
            }
        }
    }}
    run_info = get_run_info()
    for key, val in run_info.items():
        input_dict["WaveformML"]["AnalysisStep"]["_PROP_"][key] = val
    append_xml(inpath, outpath, input_dict)

if __name__ == "__main__":
    main()

