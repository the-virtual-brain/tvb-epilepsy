from collections import OrderedDict

from tvb_fit.base.utils.log_error_utils import initialize_logger

logger = initialize_logger(__name__)


# TODO: on medium term, we should remove this module.
class DictDot(object):

    def __init__(self, d):
        self.dict = dict(d)
        for key in self.dict.keys():
            if isinstance(self.dict[key], OrderedDict):
                self.dict[key] = OrderedDictDot(self.dict[key])
            elif isinstance(self.dict[key], dict):
                self.dict[key] = DictDot(self.dict[key])

    def __getattr__(self, item):
        try:
            return dict.__getattr__(self.dict, item)
        except:
            try:
                return self.dict[item]
            except:
                try:
                    for key in self.dict.keys():
                        if isinstance(key, basestring) and key.find(item) >= 0:
                            logger.warning("Item with key " + item + " not found!" +
                                    "\nReturning item with key " + key + " instead!")
                            return self.dict[key]
                except KeyError as e:
                    raise AttributeError(e)

    def __getitem__(self, item):
        return self.dict[item]

    @property
    def __dict__(self):
        return self.dict


class OrderedDictDot(object):

    def __init__(self, d):
        self.dict = dict(d)
        for key in self.dict.keys():
            if isinstance(self.dict[key], OrderedDict):
                self.dict[key] = OrderedDictDot(self.dict[key])
            elif isinstance(self.dict[key], dict):
                self.dict[key] = DictDot(self.dict[key])

    def __getattr__(self, item):
        try:
            return OrderedDict.__getattr__(self.dict, item)
        except:
            try:
                return self.dict[item]
            except:
                try:
                    for key in self.dict.keys():
                        if isinstance(key, basestring) and key.find(item) >= 0:
                            logger.warning("Item with key " + item + " not found!" +
                                    "\nReturning item with key " + key + " instead!")
                            return self.dict[key]
                except KeyError as e:
                    raise AttributeError(e)

    def __getitem__(self, item):
        return self.dict[item]

    @property
    def __dict__(self):
        return self.dict