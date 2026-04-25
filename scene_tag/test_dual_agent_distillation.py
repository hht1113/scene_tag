import sys
import unittest
from pathlib import Path
from unittest.mock import patch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from dual_agent_distillation import (
    build_annotator_system_prompt,
    build_judge_prompt,
    call_chat_completion,
    call_text_with_retries,
    extract_target_labels,
    parse_judge_output,
    parse_segments,
)


class TestExtractTargetLabels(unittest.TestCase):
    def test_extracts_target_and_definition_labels(self):
        prompt = """
TARGET LABELS (3 labels):
  Foo_Label  (中文1)
  Bar_Label  (中文2)
  not_applicable  (none)

LABEL DEFINITIONS:
1. Foo_Label (中文1):
2. Bar_Label (中文2):
"""
        labels = extract_target_labels(prompt)
        self.assertIn("Foo_Label", labels)
        self.assertIn("Bar_Label", labels)
        self.assertIn("not_applicable", labels)


class TestParseSegments(unittest.TestCase):
    def test_parse_valid_segments(self):
        text = (
            "<driving_maneuver>Foo_Label</driving_maneuver> "
            "from <start_time>0.0</start_time> to <end_time>5.0</end_time> seconds."
        )
        segments = parse_segments(text, ["Foo_Label", "not_applicable"])
        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0]["label"], "Foo_Label")
        self.assertEqual(segments[0]["start"], 0.0)
        self.assertEqual(segments[0]["end"], 5.0)


class TestParseJudgeOutput(unittest.TestCase):
    def test_parse_plain_json(self):
        raw = '{"verdict":"accepted","final_output":"x","reason":["ok"]}'
        result = parse_judge_output(raw)
        self.assertEqual(result["verdict"], "accepted")
        self.assertEqual(result["final_output"], "x")
        self.assertEqual(result["reason"], ["ok"])

    def test_parse_markdown_json(self):
        raw = """```json
{"verdict":"corrected","final_output":"y","reason":["fix"]}
```"""
        result = parse_judge_output(raw)
        self.assertEqual(result["verdict"], "corrected")
        self.assertEqual(result["final_output"], "y")

    def test_parse_fallback_text_output(self):
        raw = (
            "accepted\n"
            "<driving_maneuver>Foo_Label</driving_maneuver> "
            "from <start_time>0.0</start_time> to <end_time>5.0</end_time> seconds."
        )
        result = parse_judge_output(raw)
        self.assertEqual(result["verdict"], "accepted")
        self.assertIn("Foo_Label", result["final_output"])


class TestBuildJudgePrompt(unittest.TestCase):
    def test_contains_allowed_labels_and_draft(self):
        prompt = build_judge_prompt("draft-text", ["A_Label", "B_Label"], None)
        self.assertIn("A_Label", prompt)
        self.assertIn("B_Label", prompt)


class TestBuildAnnotatorPrompt(unittest.TestCase):
    def test_wraps_base_prompt_with_strict_rules(self):
        wrapped = build_annotator_system_prompt("BASE_PROMPT")
        self.assertIn("BASE_PROMPT", wrapped)
        self.assertIn("output ONLY the final label segments", wrapped)
        self.assertIn("Do NOT output any reasoning", wrapped)


class TestRetryHelper(unittest.TestCase):
    @patch("dual_agent_distillation.call_chat_completion")
    def test_retry_until_non_empty_text(self, mock_call):
        mock_call.side_effect = [
            ("", None),
            ("valid-output", None),
        ]
        text, error = call_text_with_retries(
            api_base="http://example/v1",
            model_name="demo",
            messages=[],
            max_tokens=128,
            temperature=0.0,
            request_timeout=30,
            max_retries=3,
        )
        self.assertEqual(text, "valid-output")
        self.assertIsNone(error)


class TestCallChatCompletionPayload(unittest.TestCase):
    @patch("dual_agent_distillation.requests.post")
    def test_response_format_passed_in_payload(self, mock_post):
        mock_resp = patch("builtins.object")
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
        resp.raise_for_status = MagicMock()
        mock_post.return_value = resp

        call_chat_completion(
            api_base="http://localhost:8000/v1",
            model_name="demo",
            messages=[],
            max_tokens=128,
            temperature=0.0,
            request_timeout=30,
            enable_thinking=False,
            response_format={"type": "json_object"},
        )
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
