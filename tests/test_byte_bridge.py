"""Tool parsing and real Qwen byte-CPU execution for the toy bridge."""

import importlib.util
import unittest

from gdn_cpu.byte_bridge import PROGRAM, parse_tool_input


class ToolParsingTests(unittest.TestCase):
    def test_valid_and_invalid_tool_calls(self) -> None:
        examples = (
            ("<tool_call><function=byte_add7><parameter=x>201</parameter>"
             "</function></tool_call>", 201),
            ("<tool_call><function=byte_add7>{\"x\":249}</function></tool_call>", 249),
            ("<tool_call><function=byte_add7><parameter=x>256</parameter>"
             "</function></tool_call>", None),
            ("<tool_call><function=calc><parameter=x>2</parameter>"
             "</function></tool_call>", None),
            ("I can calculate it myself.", None),
        )
        for text, wanted in examples:
            with self.subTest(text=text):
                self.assertEqual(parse_tool_input(text), wanted)


@unittest.skipUnless(importlib.util.find_spec("chipformer"), "Qwenputer backend unavailable")
class ByteModelToolTests(unittest.TestCase):
    def test_compiled_add7_with_wraparound(self) -> None:
        import torch

        from gdn_cpu.byte_bridge import run_cpu
        from gdn_cpu.qwen8_recurrent import build_standalone

        model, _ = build_standalone(PROGRAM)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model.to(device)
        for x, wanted in ((0, 7), (201, 208), (249, 0), (255, 6)):
            with self.subTest(x=x):
                report = run_cpu(model, x, device=device)
                self.assertEqual(report["result"], wanted)
                self.assertEqual(report["steps"], 2)


if __name__ == "__main__":
    unittest.main()
