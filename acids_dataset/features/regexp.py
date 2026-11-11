from .base import AcidsDatasetFeature
from abc import abstractmethod
import gin.torch
import logging
import re
import fnmatch


def parse_meta_regexp(meta_regexps):
    target_features = {}
    for i, m in enumerate(meta_regexps):
        features_tmp = re.findall(r'{{\w+?}}', m)
        if len(features_tmp) > 0:
            for f in features_tmp: 
                f_name = re.match(r"{{(\w+)}}", f).groups()[0]
                target_features[f_name] = target_features.get(f_name, []) + [i]
        else:
            logging.warning('regexp %s provided no valid feature.'%m)
    return [RegexpFeature(*[meta_regexps[i] for i in v], name=k) for k, v in target_features.items()]

def append_meta_regexp(features, meta_regexp):
    return features + parse_meta_regexp(meta_regexp)



def glob_to_regex(glob_pattern, feature_name):
    """
    Convert a glob pattern with extraction like **/music/*_{{idx}}.wav to a regex pattern.
    """
    # Escape special characters in regex, except for the {} which indicates variables
    regex_pattern = re.escape(glob_pattern)

    # Replace the escaped wildcards and {{idx}} with appropriate regex patterns
    regex_pattern = regex_pattern.replace(r'\*\*/', r'(?:.*/)')  # **/ -> match any path
    regex_pattern = regex_pattern.replace(r'\*', r'[^/]*')       # * -> match any except '/'
    
    # Specify the capture group for variables like {{idx}}
    regex_pattern = re.sub(r'\\{\\{%s\\}\\}'%feature_name, r'(\\w+)?', regex_pattern)
    regex_pattern = re.sub(r'\\{\\{\w+\\}\\}', r'\\w*?', regex_pattern)
    
    # Ensure it matches the whole line
    regex_pattern = f'^{regex_pattern}$'
    
    return regex_pattern

@gin.configurable(module="ad.features")
class RegexpFeature(AcidsDatasetFeature):
    has_hash = True

    def __init__(self, *regexps, name=None, **kwargs):
        self.regexps = regexps
        assert name is not None, "name is required for RegexpFeature"
        super().__init__(name=name, **kwargs)

    def __repr__(self):
        return f"RegexpFeature(name=f{self.feature_name}, regexps={self.regexps})"

    @property
    def default_feature_name(self):
        return self.feature_name

    def _decide_among_candidates(self, candidates):
        return candidates[0]

    def _extract_id_from_path(self, path):
        meta_candidates = []
        for r in self.regexps:
            op_regexp = glob_to_regex(r, self.feature_name)
            re_result = re.match(op_regexp, path)
            if re_result is not None:
                for i, rx in enumerate(re_result.groups()):
                    meta_candidates.append(rx)
        if len(meta_candidates) == 0:
            return None
        elif len(meta_candidates) == 1:
            return meta_candidates[0]
        elif len(meta_candidates) > 2:
            meta = self._decide_among_candidates(meta_candidates)
            logging.warning(f'{self} : found several for path {path}. Taking {meta}')
            return meta
        return meta_candidates
    
    def from_fragment(self, fragment, write: bool = True):
        audio_path = fragment.get_metadata()['audio_path']
        meta_id = self._extract_id_from_path(audio_path)
        if write:
            fragment.update_metadata(**{self.feature_name: meta_id})
        return meta_id

    def read(self, fragment):
        return fragment.get_metadata().get(self.feature_name)

    @abstractmethod
    def __call__(self, x):
        raise NotImplementedError
        
        
