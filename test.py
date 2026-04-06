import json


res = json.loads('{"prompt": "Replace all numbers in \'Hello 34 I\'m 233 years old\' with NUMBERS", "function": "fn_substitute_string_with_regex", "arguments": {"source_string":  "Hello 34 I\'m 233 years old","regex":  "([0-9]+)","replacement":  "NUMBERS"}}')