"""Synthetic stdio peer for output-schema tests; never opens URLs or files."""

import json
import sys


def main() -> None:
    reference = sys.argv[1]
    for line in sys.stdin:
        request = json.loads(line)
        if "id" not in request:
            continue
        method = request["method"]
        if method == "initialize":
            result = {
                "protocolVersion": request["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "schema-fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [{
                "name": "sample",
                "inputSchema": {"type": "object", "properties": {}},
                "outputSchema": {
                    "type": "object",
                    "$defs": {"value": {"type": "string"}},
                    "properties": {"value": {"$ref": reference}},
                    "required": ["value"],
                },
            }]}
        elif method == "tools/call":
            result = {"content": [], "structuredContent": {"value": "synthetic"}}
        else:
            raise AssertionError(f"Unexpected fixture method: {method}")
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)


if __name__ == "__main__":
    main()
