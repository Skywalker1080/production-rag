"""Smoke tests for the centralized logger (stdlib unittest, no new deps)."""

import io
import json
import unittest

from production_rag.common import logging as plog


def _lines(buf: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in buf.getvalue().splitlines()]


class LoggingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.buf = io.StringIO()
        plog.configure(level="DEBUG", stream=self.buf, force=True)

    def test_json_shape_and_context(self) -> None:
        plog.new_request_id()
        logger = plog.get_logger("test.shape")
        logger.info("hello")
        (line,) = _lines(self.buf)
        self.assertEqual(line["level"], "INFO")
        self.assertEqual(line["message"], "hello")
        self.assertIsNotNone(line["request_id"])
        self.assertIn("timestamp", line)

    def test_stage_emits_enter_exit_and_binds_stage(self) -> None:
        logger = plog.get_logger("test.stage")
        with plog.stage("identifier"):
            logger.info("inside")
        lines = _lines(self.buf)
        self.assertEqual(
            [line["message"] for line in lines],
            ["stage_enter", "inside", "stage_exit"],
        )
        self.assertTrue(all(line["stage"] == "identifier" for line in lines))
        exit_line = lines[-1]
        self.assertIn("duration_ms", exit_line)

    def test_component_bound_to_stage_and_reset_after(self) -> None:
        logger = plog.get_logger("test.component")
        logger.info("outside")
        with plog.stage("identifier", component="indexing"):
            logger.info("inside")
        logger.info("after")
        lines = _lines(self.buf)
        self.assertIsNone(lines[0]["component"])
        self.assertEqual(lines[1]["component"], "indexing")
        self.assertEqual(lines[1]["stage"], "identifier")
        self.assertIsNone(lines[-1]["component"])

    def test_stage_error_names_component(self) -> None:
        with self.assertRaises(ValueError):
            with plog.stage("normalizer", component="indexing"):
                raise ValueError("boom")
        lines = _lines(self.buf)
        error_lines = [line for line in lines if "stage_error" in line["message"]]
        self.assertEqual(len(error_lines), 1)
        self.assertEqual(error_lines[0]["component"], "indexing")
        self.assertEqual(error_lines[0]["stage"], "normalizer")

    def test_log_stage_decorator_with_component(self) -> None:
        @plog.log_stage(component="indexing")
        def identifier() -> str:
            return "ok"

        self.assertEqual(identifier(), "ok")
        lines = _lines(self.buf)
        self.assertEqual(
            [line["message"] for line in lines],
            ["stage_enter", "stage_exit"],
        )
        self.assertTrue(all(line["component"] == "indexing" for line in lines))

    def test_stage_error_marks_breakpoint_and_reraises(self) -> None:
        logger = plog.get_logger("test.error")
        with self.assertRaises(ValueError):
            with plog.stage("normalizer"):
                logger.info("before boom")
                raise ValueError("boom")
        lines = _lines(self.buf)
        error_lines = [line for line in lines if "stage_error" in line["message"]]
        self.assertEqual(len(error_lines), 1)
        self.assertEqual(error_lines[0]["level"], "ERROR")
        self.assertIn("traceback", error_lines[0])
        self.assertIn("ValueError", error_lines[0]["traceback"])


if __name__ == "__main__":
    unittest.main()
