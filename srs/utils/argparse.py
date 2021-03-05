import argparse


class ArgumentParser(argparse.ArgumentParser):
    def __init__(self, **kwargs):
        super().__init__(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter, **kwargs
        )
        self.optional = self._action_groups.pop()
        self.required = self.add_argument_group('required arguments')
        self._action_groups.append(self.optional)

    def add_argument(self, *args, **kwargs):
        if kwargs.get('required', False):
            return self.required.add_argument(*args, **kwargs)
        else:
            return super().add_argument(*args, **kwargs)
