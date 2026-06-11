import json

def _parse_json(content: str):
    if not content:
        return {}
    s = content.strip()
    if s.startswith("```"):
        first_newline = s.find("\n")
        if first_newline != -1:
            s = s[first_newline:].strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        if "Unterminated string" in str(e) or "Expecting" in str(e):
            for closure in ['"', '"}', '"]}', '"]} }', '"} }', '"} } }']:
                try:
                    return json.loads(s + closure)
                except json.JSONDecodeError:
                    pass
            
            last_quote = s.rfind('"')
            if last_quote > 0:
                s_chopped = s[:last_quote].strip()
                if s_chopped.endswith(','):
                    s_chopped = s_chopped[:-1]
                elif s_chopped.endswith(':'):
                    s_chopped += ' null'
                
                for closure in ['}', ']}', '}]}', '}}', '}}}']:
                    try:
                        return json.loads(s_chopped + closure)
                    except json.JSONDecodeError:
                        pass
        raise e

content = """{
  "intent": "bias_detection",
  "audit_narrative": {
    "executive_summary": "This audit assessed potential bias in the 'energy_source' dataset across different 'region' groups using a proxy model. No significant bias was detected, with all fairness metrics"""
print(_parse_json(content))
