import json
import ruamel.yaml
from ruamel.yaml import YAMLError
from ruamel.yaml.comments import CommentedMap
from ruamel.yaml.scalarstring import PreservedScalarString, SingleQuotedScalarString
from ruamel.yaml.compat import string_types, MutableMapping, MutableSequence
import os
from src.utils.util import OrderlyJSONEncoder


class JSONYAML:

    def __init__(self, seq=None, offset=None):
        self.yaml = ruamel.yaml.YAML()  # this uses the new API
        # if you have standard indentation, no need to use the following
        params = {}
        if seq:
            params["sequence"] = seq
        if offset:
            params["offset"] = offset
        self.yaml.indent(**params)

    def yaml_to_json(self, in_file, out_file):
        with open(in_file, 'r') as stream:
            try:
                data_map = self.yaml.load(stream)
                with open(out_file, 'w') as output:
                    output.write(OrderlyJSONEncoder(indent=2).encode(data_map))
            except YAMLError as exc:
                print(exc)
                return False
        return True

    def json_to_yaml(self, in_file, out_file, restore_scalars=False):
        with open(in_file, 'r') as stream:
            try:
                data_map = json.load(stream, object_pairs_hook=CommentedMap)
                # if you need to "restore" literal style scalars, etc.
                if restore_scalars:
                    JSONYAML.walk_tree(data_map)
                with open(out_file, 'w') as output:
                    self.yaml.dump(data_map, output)
            except self.yaml.YAMLError as exc:
                print(exc)
                return False
        return True

    @staticmethod
    def preserve_literal(s):
        return PreservedScalarString(s.replace('\r\n', '\n').replace('\r', '\n'))

    @staticmethod
    def walk_tree(base):
        if isinstance(base, MutableMapping):
            for k in base:
                v = base[k]  # type: Text
                if isinstance(v, string_types):
                    if '\n' in v:
                        base[k] = JSONYAML.preserve_literal(v)
                    elif '${' in v or ':' in v:
                        base[k] = SingleQuotedScalarString(v)
                else:
                    JSONYAML.walk_tree(v)
        elif isinstance(base, MutableSequence):
            for idx, elem in enumerate(base):
                if isinstance(elem, string_types):
                    if '\n' in elem:
                        base[idx] = JSONYAML.preserve_literal(elem)
                    elif '${' in elem or ':' in elem:
                        base[idx] = SingleQuotedScalarString(elem)
                else:
                    JSONYAML.walk_tree(elem)



def main():
    def convert_file(p, jy):
        if p.endswith(".json"):
            if jtoy:
                jy.json_to_yaml(p,p[0:-5]+".yaml")
            else:
                raise IOError("You specified yaml to json conversion but gave a json file. Use -y to convert to yaml.".format(p))
        elif p.endswith(".yaml"):
            if ytoj:
                jy.yaml_to_json(p,p[0:-5]+".json")
            else:
                raise IOError("You specified json to yaml conversion but gave a yaml file. Use -j to convert to json.".format(p))
        else:
            raise IOError("The specified file {} is not a json file must end with .json".format(p))

    import argparse
    import sys
    from pathlib import Path
    argparse = argparse.ArgumentParser()
    argparse.add_argument("path",help="path to file or folder of .json or .yaml files.",type=str)
    argparse.add_argument("--jsontoyaml", "-y", action="store_true", help="Convert found json files to yaml.")
    argparse.add_argument("--yamltojson", "-j", action="store_true", help="Convert found yaml files to json.")
    argparse.add_argument("--recursive", "-r", action="store_true", help="Recurse into directories.")
    args = argparse.parse_args()
    jtoy = False
    ytoj = False
    jy = JSONYAML()
    if args.yamltojson:
        ytoj = True
    if args.jsontoyaml:
        jtoy = True
    if (not jtoy) and (not ytoj):
        print("specify -y to convert to yaml or -j to convert to json.")
        sys.exit()
    args.path = os.path.normpath(os.path.abspath(args.path))
    if os.path.isfile(args.path):
        convert_file(args.path, jy)
    else:
        p = Path(args.path)
        ext = ".json" if jtoy else ".yaml"
        if args.recursive:
            files = sorted(p.glob('**/*.{0}'.format(ext)))
        else:
            files = sorted(p.glob(ext))
        for f in files:
            convert_file(str(f.resolve()), jy)

