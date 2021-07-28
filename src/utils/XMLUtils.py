import xml.etree.ElementTree as ET
from os.path import exists
from src.utils.util import get_run_info, get_file_md5, check_path
from os.path import expanduser
import sys
from ntpath import basename

class XMLWriter:
    def __init__(self):
        self.step_xml = {}
        self.code = basename(str(sys.argv[0]))
        self.input_file = "UNKNOWN"
        self.output_file = "UNKNOWN"
        self.step_name = "UNKNOWN"
        self.step_settings = {}

    def generate_step_xml(self, runtime):
        self.step_xml = { "AnalysisStep": {
            "_PROP_": {
                "code": self.code
            },
            "input": {
                "_PROP_": {
                    "file": self.input_file,
                    "md5": get_file_md5(self.input_file)
                }
            },
            "output": {
                "_PROP_": {
                    "file": self.output_file
                }
            },
            self.step_name: {
                "_PROP_": self.step_settings
            }
        }}
        run_info = get_run_info()
        for key, val in run_info.items():
            self.step_xml["AnalysisStep"]["_PROP_"][key] = val
        self.step_xml["AnalysisStep"]["_PROP_"]["dtime"] = str(int(runtime))

    def write_xml(self, out_path, runtime):
        """
        writes xml to file at out_path
        @param out_path: string to .xml file output
        @param runtime: time in seconds of program execution
        @return:
        """
        self.generate_step_xml(runtime)
        if exists(self.input_file):
            append_xml(self.input_file, out_path, self.step_xml)
        else:
            print("No input XML file {} found, skipping".format(self.input_file))


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
                root.set(key, str(append_dict[name][key]))
            continue
        n = ET.Element(name)
        if isinstance(append_dict[name] , dict):
            append_xml(in_path, out_path, append_dict[name], n)
        else:
            n.text = str(append_dict[name])
        root.append(n)
    if parent is None:
        _pretty_print(root)
        tree.write(out_path, xml_declaration=True)

def _pretty_print(current, parent=None, index=-1, depth=0):
    for i, node in enumerate(current):
        _pretty_print(node, current, i, depth + 1)
    if parent is not None:
        if index == 0:
            parent.text = '\n' + ('    ' * depth)
        else:
            parent[index - 1].tail = '\n' + ('    ' * depth)
        if index == len(parent) - 1:
            current.tail = '\n' + ('    ' * (depth - 1))

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

