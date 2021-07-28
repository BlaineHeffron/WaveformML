import h5py


class H5FileHandler(h5py.File):
    def __init__(self, path, *args, **kwargs):
        try:
            super(H5FileHandler, self).__init__(path, *args, **kwargs)
        except Exception as e:
            print("Opening {} failed:\n".format(path))
            raise e
