import math

import yaml

yaml.add_constructor(
    "!product",
    lambda loader, node: math.prod(loader.construct_sequence(node)),
    Loader=yaml.SafeLoader,
)


class NamedDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]
        else:
            return None

    def __setattr__(self, key, value):
        self[key] = value

    def to_dict(self):
        out = {}
        for k, v in self.items():
            if isinstance(v, NamedDict):
                out[k] = v.to_dict()
            else:
                out[k] = v
        return out

    @classmethod
    def from_dict(cls, d):
        out = cls()
        for k, v in d.items():
            if isinstance(v, dict):
                out[k] = cls().from_dict(v)
            else:
                out[k] = v
        return out

    @classmethod
    def from_yaml(cls, path, safe=True):
        with open(path, "r") as f:
            out = NamedDict.from_dict(yaml.safe_load(f) if safe else yaml.full_load(f))
        return out

    def to_yaml(self, path):
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_hparam(self):
        def flatten_dictionary(nested_dict, parent_key="", sep="."):
            items = []
            for key, value in nested_dict.items():
                new_key = parent_key + sep + key if parent_key else key
                if isinstance(value, dict):
                    items.extend(flatten_dictionary(value, new_key, sep=sep).items())
                else:
                    items.append((new_key, str(value)))
            return dict(items)
        return flatten_dictionary(self.to_dict())
