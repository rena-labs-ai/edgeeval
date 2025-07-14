import dpkt
import json
from typing import List, Dict, Any, Optional
from pathlib import Path


def ollama_get_tokens_breakdown(response, tokenizer):
    """
    response:
        {'model': 'qwen2.5:14b',
        'created_at': '2025-07-09T18:02:49.880726274Z',
        'message': {'role': 'assistant',
        'content': "It appears there's an ongoing issue with the search service due to an invalid subscription token. I'll try a different approach by manually navigating to the official Axelar website and scraping the information about their core team and founders from there.\n\nLet me first install the necessary browser and then proceed to access the Axelar site.\n\n\n",
        'tool_calls': [{'function': {'name': 'browser_install', 'arguments': {}}},
        {'function': {'index': 1,
            'name': 'browser_navigate',
            'arguments': {'url': 'https://axelar.network'}}}]},
        'done_reason': 'stop',
        'done': True,
        'total_duration': 3980230924,
        'load_duration': 126967212,
        'prompt_eval_count': 2787,
        'prompt_eval_duration': 137746030,
        'eval_count': 105,
        'eval_duration': 2880956856}
    tokenizer:
        AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B")
    return: {
        'tool_call_tokens': tool_calls_tokens_count,
        'non_tool_call_tokens': total_tokens_count - tool_calls_tokens_count,
        'total_tokens': total_tokens_count
    }
    """

    def ollama_get_tool_calls_tokens(output_idx: List[int]):
        """
        151657: <tool_call>
        151658: </tool_call>
        """
        tool_call_token_id = 151657
        end_tool_call_token_id = 151658
        tool_call_tokens_count = 0
        is_in_tool_call = False
        for idx in output_idx:
            if idx == tool_call_token_id:
                is_in_tool_call = True
                tool_call_tokens_count += 1
            elif idx == end_tool_call_token_id:
                is_in_tool_call = False
                tool_call_tokens_count += 1
            elif is_in_tool_call:
                tool_call_tokens_count += 1
        return tool_call_tokens_count
    
    def serialize_json_to_string(json_data):
        """
        json_data:
            {
            "role":"assistant",
            "content":"<think>\nOkay, the user wants me to say hello to Bob. Let me check the tools provided. There\\'s a function called say_hello that takes a name parameter. The required parameter is name, and it\\'s a string. So I need to call say_hello with the argument \"Bob\". I\\'ll make sure to format the tool call correctly in JSON inside the XML tags.\n</think>\n\n",
            "tool_calls":[
                {
                    "function":{
                        "name":"say_hello",
                        "arguments":{
                        "name":"Bob"
                        }
                    }
                },
                {
                    "function":{
                        "name":"another_function",
                        "arguments":{
                        "param":"value"
                        }
                    }
                }
            ]
            }
        """

        formatted_string = json_data['content'].strip() + '\n'
        if 'tool_calls' in json_data:
            for tool_call in json_data['tool_calls']:
                formatted_string += '<tool_call>\n'
                tool_call_json = json.dumps(tool_call['function'])
                formatted_string += tool_call_json + '\n'
                formatted_string += '</tool_call>'
        return formatted_string
    
    formatted_string = serialize_json_to_string(response['message'])
    output_ids = tokenizer(formatted_string, return_tensors="pt").input_ids[0].tolist()
    print(f"{formatted_string}")

    tool_call_tokens_count = ollama_get_tool_calls_tokens(output_ids)

    return {
        'tool_call_tokens': tool_call_tokens_count,
        'non_tool_call_tokens': len(output_ids) - tool_call_tokens_count,
        'total_tokens': len(output_ids)
    }


class Ollama_pcap_iter:
    """
    return: response
    """

    def __init__(self, pcap_file):
        SERVER_PORT = 11434

        f = open(pcap_file, 'rb')
        pcap = dpkt.pcap.Reader(f)

        responses = []

        for ts, buf in pcap:
            try:
                eth = dpkt.ethernet.Ethernet(buf)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue

                ip = eth.data
                if not isinstance(ip.data, dpkt.tcp.TCP):
                    continue

                tcp = ip.data
                if tcp.dport == SERVER_PORT:
                    continue  # client -> server
                if tcp.sport != SERVER_PORT:
                    continue  # not server -> client

                if len(tcp.data) == 0:
                    continue

                try:
                    http = dpkt.http.Response(tcp.data)
                    body = http.body.strip()
                    if body.startswith(b'{') or body.startswith(b'['):
                        obj = json.loads(body.decode(errors='ignore'))
                        responses.append(obj)
                except (dpkt.UnpackError, ValueError):
                    continue

            except Exception:
                continue

        f.close()

        self.responses = responses
        self.idx = 0
        self.end = len(responses)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.end:
            raise StopIteration
        response = self.responses[self.idx]
        self.idx += 1
        return response

    def __len__(self):
        return self.end

    def reset(self):
        self.idx = 0


import re
import json
from pathlib import Path
from typing import Union, Dict, Any

# --------------------------------------------------------------------------- #
#  Regexes reused inside the function (compiled once to avoid recomputation)  #
# --------------------------------------------------------------------------- #
_TIMESTAMP_LINE = re.compile(
    r'^\x1b\[2m\d{4}-\d{2}-\d{2}T.*? INFO.*$', re.MULTILINE
)
_ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
_JSON_BLOCK = re.compile(
    r'\{\s*"total_tests"\s*:[\s\S]*?\}\s*$', re.DOTALL
)


def extract_json_from_trace(
    log_file: Union[str, Path],
    *,
    json_regex: re.Pattern = _JSON_BLOCK
) -> Dict[str, Any]:
    """
    Extract the central JSON payload from a trace log and return it as a dict.

    Parameters
    ----------
    log_file : str | Path
        Path to the trace log (text file).
    json_regex : re.Pattern, optional
        Pattern that identifies the JSON block.  By default, it assumes the
        block begins with "total_tests".  Override if your format changes.

    Returns
    -------
    Dict[str, Any]
        The parsed JSON object.

    Raises
    ------
    RuntimeError
        If no JSON block matching `json_regex` is found.
    json.JSONDecodeError
        If the extracted block is not valid JSON.
    """
    # 1 — read the entire file (UTF-8, forgiving on errors)
    text = Path(log_file).read_text(encoding="utf-8", errors="ignore")

    # 2 — strip timestamp lines and any other lingering ANSI codes
    text = _TIMESTAMP_LINE.sub("", text)
    text = _ANSI_ESCAPE.sub("", text)

    # 3 — locate the JSON payload
    match = json_regex.search(text)
    if not match:
        raise RuntimeError("JSON block not found – adjust `json_regex`.")

    # 4 — parse and return
    return json.loads(match.group(0))


_METRIC_RE = re.compile(
    r"""input_tokens:\s*(?P<prefill_tokens>\d+).*?
        output_tokens:\s*(?P<decode_tokens>\d+).*?
        latency:\s*load_model:\s*(?P<load_model>\d+).*?
        prefill:\s*(?P<prefill>\d+).*?
        decode:\s*(?P<decode>\d+).*?
        (?:tool_call:\s*(?P<tool_call>\d+))?   # optional
    """,
    re.S | re.X,
)

def parse_step_metrics(log: str):
    """Extracts StepMetrics from a usage/latency log line."""
    match = _METRIC_RE.search(log)
    if not match:
        raise ValueError("Expected metric fields not found.")

    gd = match.groupdict()

    time_model_load = float(gd["load_model"])
    prefill_time   = float(gd["prefill"])
    decode_time    = float(gd["decode"])
    time_llm_call  = prefill_time + decode_time
    prefill_tokens = int(gd["prefill_tokens"])
    decode_tokens = int(gd["decode_tokens"])

    return time_model_load, \
           time_llm_call, \
           prefill_time, \
           decode_time, \
           prefill_tokens, \
           decode_tokens


class Ollama_trace_iter:
    """
    Iterator over trace JSON to yield:
    - {'create_execution_plan': {latency: xxx}}
    - {'execute_execution_plan': {'latency': xxx}}
    """

    def __init__(self, trace_path, results_idx=0, runs_idx=0):
        """
        Initializes the iterator by loading and parsing the trace JSON file.

        Args:
            trace_path (str): The path to the JSON trace file.
        """
        try:
            trace_data = extract_json_from_trace(trace_path)
            # print(f"trace_data: {trace_data}")
            # Navigate to the list of trace steps
            self.steps = trace_data['results'][results_idx]['runs'][runs_idx]['trace']['inner_traces']
        except (IOError, json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error loading or parsing trace file: {e}")
            self.steps = []

        self.index = 0

    def __iter__(self):
        """Returns the iterator object itself."""
        return self

    def __next__(self):
        """
        Yields the next relevant log entry from the trace.

        Raises:
            StopIteration: When all trace steps have been processed.

        Returns:
            dict: The next log entry.
        """
        while self.index < len(self.steps):
            step = self.steps[self.index]
            self.index += 1  # Increment index for the next call

            label = step.get('label')

            if label == 'create_execution_plan':
                return {'create_execution_plan': {'latency': step.get('latency')}}

            if label == 'execute_execution_plan':
                return {'execute_execution_plan': {'latency': step.get('latency')}}

        # If the loop finishes, there are no more steps to process
        raise StopIteration
