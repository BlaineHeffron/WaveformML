import h5py


class H5FileHandler(h5py.File):
    def __init__(self, path, *args, **kwargs):
        try:
            super().__init__(path, *args, **kwargs)
        except Exception as e:
            print("Opening {} failed:\n".format(path))
            print(e)
