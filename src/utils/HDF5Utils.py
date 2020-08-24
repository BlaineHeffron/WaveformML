import h5py


class H5FileHandler:
    def __init__(self, path, *args, **kwargs):
        try:
            self.h5f = h5py.File(path, *args, **kwargs)
        except Exception as e:
            print("Opening {} failed:\n".format(path))
            raise e

    def __iter__(self, item):
        return self.h5f.__iter__(item)

    def __getitem__(self, name):
        return self.h5f.__getitem__(name)

    def __getattr__(self, attr):
        if attr == "h5f":
            return self.h5f
        return self.h5f[attr]

    def __enter__(self, *args, **kwargs):
        return self.h5f

    def __exit__(self, *args, **kwargs):
        self.h5f.__exit__(*args, **kwargs)
