from src.wavepackets.grammar import parse, emit_json_trace
import json


def test_parse_and_emit(tmp_path):
    program = """
    T1 F1 F2
    T2
    T3 F3 # comment
    """
    instrs = parse(program)
    assert instrs[0].op == "T1"
    assert instrs[0].flags == ["F1", "F2"]
    path = tmp_path / "trace.json"
    emit_json_trace(instrs, path)
    data = json.loads(path.read_text())
    assert data[2]["op"] == "T3" and data[2]["flags"] == ["F3"]
