import importlib
import os


class DictionaryUtility:
    """
    Utility methods for dealing with dictionaries.
    """

    @staticmethod
    def to_object(item):
        """
        Convert a dictionary to an object (recursive).
        """

        def convert(item):
            if isinstance(item, dict):
                return type('jo', (), {k: convert(v) for k, v in item.items()})
            if isinstance(item, list):
                def yield_convert(item):
                    for index, value in enumerate(item):
                        yield convert(value)

                return list(yield_convert(item))
            else:
                return item

        return convert(item)

    def to_dict(obj):
        """
         Convert an object to a dictionary (recursive).
         """

        def convert(obj):
            if not hasattr(obj, "__dict__"):
                return obj
            result = {}
            for key, val in obj.__dict__.items():
                if key.startswith("_"):
                    continue
                element = []
                if isinstance(val, list):
                    for item in val:
                        element.append(convert(item))
                else:
                    element = convert(val)
                result[key] = element
            return result

        return convert(obj)


class ModuleUtility:
    """
    Utility methods for loading modules and chaining their functions
    """

    def __init__(self, modlist):
        self.modules = {}
        self.classes = {}
        self.modList = modlist
        modNames = [m.split('.')[-1] for m in modlist]
        for i, m in enumerate(modlist):
            self.modules[modNames[i]] = importlib.import_module(m)

    def retrieve_module(self, name):
        if name in self.modules:
            return self.modules[name]
        else:
            raise IOError("{0} is not in the module list.".format(name))

    def retrieve_class(self, class_string):
        if class_string in self.classes:
            return self.classes[class_string]
        if "." in class_string:
            info = class_string.split(".")
            self.classes[class_string] = getattr(self.retrieve_module(info[0]), info[1])
            return self.classes[class_string]
        else:
            raise IOError("{0} is not a valid class path.\n"
                          " Must be formatted <module name>.<class name>".format(class_string))

    def create_class_instances(self, classes):
        instances = []
        last_type = None
        prev_object = None
        for class_name in classes:
            if isinstance(class_name, str):
                if last_type is str:
                    instances.append(self.retrieve_class(prev_object))
                prev_object = class_name
                last_type = str
            elif isinstance(class_name, list):
                if last_type is str:
                    instances.append(self.retrieve_class(prev_object)(*class_name))
                else:
                    raise IOError("Argument list must be preceded by a string of the class path.\n"
                                  "Errored at input: ", str(class_name))
                last_type = list
                prev_object = None
            elif isinstance(class_name, dict):
                if len(class_name.keys()) > 1:
                    print("Warning: Multiple keyed dictionaries might not get entered in proper order.\n"
                          " Please use an array of dictionaries with a single key instead to specify the object. \n"
                          " Or a list alternating key (string) arguments (list) pairs.")
                for key in class_name:
                    instances.append(self.retrieve_class(class_name[key])(**classes[class_name[key]]))
                last_type = dict
                prev_object = None
        if last_type is str:
            instances.append(self.retrieve_class(prev_object))
        return instances


def path_split(a_path):
    return os.path.normpath(a_path).split(os.path.sep)

def path_create(a_path):
    if not os.path.exists(a_path):
        os.mkdir(os.path.expanduser(os.path.normpath(a_path)))

def unique_path_combine(pathlist):
    common = []
    i = 1
    for path in pathlist:
        path_array = path_split(path)
        if common:
            while common[0:i] == path_array[0:i]:
                i += 1
            common = common[0:i - 1]
        else:
            common = path_array
    output_string = ""
    if len(common) > 0:
        for path in pathlist:
            path_array = path_split(path)
            i = 0
            while path_array[i] == common[i]:
                i += 1
            if output_string != "":
                output_string += "__{0}".format('_'.join(path_array[i:]))
            else:
                output_string = '_'.join(path_array[i:])
    else:
        for path in pathlist:
            path_array = path_split(path)
            if output_string != "":
                output_string += "__{0}".format('_'.join(path_array))
            else:
                output_string = '_'.join(path_array)
    return output_string
