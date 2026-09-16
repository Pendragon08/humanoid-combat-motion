"""Caption-only weak conditions, matching the existing training vocabulary."""
import re

PATTERNS = {
    'punch': r'\bpunch(?:es|ing)?\b|\bfists?\b',
    'kick': r'\bkick(?:s|ing)?\b',
    'boxing': r'\bbox(?:ing|ers?)\b',
    'martial_arts': r'\bmartial arts\b|\bkung fu\b',
    'defense': r'\bdodg(?:e|es|ing)\b|\bdefen(?:d|ds|ding|se)\b|\bblock(?:s|ing)?\b',
}


def caption_conditions(text):
    return {key: int(re.search(pattern, text, re.I) is not None)
            for key, pattern in PATTERNS.items()}


def assert_no_test_overlap(provenance, test_ids):
    used = {source for kind in ('training_sources', 'validation_sources')
            for group in provenance[kind] for source in group}
    overlap = used & {'interhuman:' + str(x) for x in test_ids}
    if overlap:
        raise ValueError('Adapter training/selection overlaps test: ' + ','.join(sorted(overlap)))
