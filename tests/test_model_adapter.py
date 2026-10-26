import json
import unittest

from src.model_adapter import (
    ModelConfig,
    ModelNotConfiguredError,
    ModelRequestError,
    OpenAIResponsesAdapter,
)


class FakeResponse:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.value, ensure_ascii=False).encode("utf-8")


class FakeOpener:
    def __init__(self, value):
        self.value = value
        self.calls = []

    def __call__(self, request, *, timeout):
        self.calls.append((request, timeout))
        return FakeResponse(self.value)


class OpenAIResponsesAdapterTests(unittest.TestCase):
    def test_posts_responses_json_mode_and_collects_text_from_all_message_items(self):
        opener = FakeOpener({
            "status": "completed",
            "usage": {"input_tokens": 11, "output_tokens": 3, "total_tokens": 14},
            "output": [
                {"type": "reasoning", "summary": []},
                {"type": "message", "content": [
                    {"type": "output_text", "text": "{\"candidates\":"},
                    {"type": "output_text", "text": "[]}"},
                ]},
            ],
        })
        config = ModelConfig(
            model="configured-model", api_key="secret-key",
            base_url="https://example.invalid/v1/", timeout_seconds=12,
        )
        adapter = OpenAIResponsesAdapter(config, opener=opener)
        self.assertEqual('{"candidates":[]}', adapter.complete_json(
            task="extract", payload={"message": "中文"},
        ))
        self.assertEqual({"input_tokens": 11, "output_tokens": 3}, adapter.last_usage)
        request, timeout = opener.calls[0]
        self.assertEqual("https://example.invalid/v1/responses", request.full_url)
        self.assertEqual(12.0, timeout)
        self.assertEqual("Bearer secret-key", request.headers["Authorization"])
        body = json.loads(request.data)
        self.assertFalse(body["store"])
        self.assertEqual({"type": "json_object"}, body["text"]["format"])
        self.assertIn("JSON", body["input"][0]["content"])
        self.assertEqual("中文", json.loads(body["input"][1]["content"])["payload"]["message"])
        self.assertNotIn("secret-key", repr(config))
        opener.value = {"status": "completed", "output_text": '{"candidates":[]}'}
        adapter.complete_json(task="extract", payload={"message": "again"})
        self.assertEqual({}, adapter.last_usage)

    def test_posts_chat_completions_json_mode_and_maps_usage(self):
        opener = FakeOpener({
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": '{"answer":"ok"}'},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
        })
        adapter = OpenAIResponsesAdapter(ModelConfig(
            model="configured-model",
            api_key="secret-key",
            base_url="https://api.siliconflow.cn/v1/",
            timeout_seconds=30,
            api_mode="chat_completions",
            max_output_tokens=256,
        ), opener=opener)

        self.assertEqual('{"answer":"ok"}', adapter.complete_json(
            task="benchmark_answer", payload={"question": "哪里？"},
        ))
        self.assertEqual({"input_tokens": 12, "output_tokens": 4}, adapter.last_usage)
        request, timeout = opener.calls[0]
        self.assertEqual(
            "https://api.siliconflow.cn/v1/chat/completions", request.full_url,
        )
        self.assertEqual(30.0, timeout)
        body = json.loads(request.data)
        self.assertEqual(["system", "user"], [item["role"] for item in body["messages"]])
        self.assertEqual({"type": "json_object"}, body["response_format"])
        self.assertFalse(body["stream"])
        self.assertEqual(256, body["max_tokens"])
        self.assertFalse({"store", "input", "text"} & set(body))

        opener.value = {
            "choices": [{
                "message": {"content": '{"answer":"cut off"}'},
                "finish_reason": "length",
            }],
            "usage": {"prompt_tokens": 20, "completion_tokens": 256},
        }
        with self.assertRaises(ModelRequestError):
            adapter.complete_json(task="benchmark_answer", payload={})
        self.assertEqual({}, adapter.last_usage)

    def test_blank_key_or_model_fails_before_opening_network(self):
        opener = FakeOpener({})
        for config in (
            ModelConfig(model="", api_key="key"),
            ModelConfig(model="model", api_key=""),
        ):
            with self.assertRaises(ModelNotConfiguredError):
                OpenAIResponsesAdapter(config, opener=opener).complete_json(task="x", payload={})
        with self.assertRaises(ValueError):
            OpenAIResponsesAdapter(ModelConfig(
                model="model", api_key="key", api_mode="unknown",
            ), opener=opener).complete_json(task="x", payload={})
        self.assertEqual([], opener.calls)

    def test_incomplete_refusal_and_malformed_envelope_are_explicit_errors(self):
        values = [
            {"status": "incomplete", "output": []},
            {"status": "completed", "output": [{"type": "message", "content": [
                {"type": "refusal", "refusal": "no"},
            ]}]},
            {"status": "completed", "output": "wrong"},
        ]
        for value in values:
            adapter = OpenAIResponsesAdapter(
                ModelConfig(model="m", api_key="k"), opener=FakeOpener(value),
            )
            with self.assertRaises(ModelRequestError):
                adapter.complete_json(task="x", payload={})


if __name__ == "__main__":
    unittest.main()
