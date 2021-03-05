class Dict(dict):
    def __getattr__(self, key):
        return self.__getitem__(key)

    def __setattr__(self, key, val):
        return self.__setitem__(key, val)

    def __setitem__(self, key, val):
        if type(val) is dict:
            val = Dict(val)
        super().__setitem__(key, val)

    def __str__(self):
        import re

        _str = ''
        for key, val in sorted(self.items()):
            if type(key) is not str:
                continue
            val_str = str(val)
            if len(val_str) > 80:
                val_str = re.sub(r'\s+', ' ', val_str.strip())[:60]
                val_str = f'"{val_str}..."'
            _str += f"{key}: {val_str}\n"
        return _str
