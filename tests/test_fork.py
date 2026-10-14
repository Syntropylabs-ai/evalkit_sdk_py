"""A process forked after init() (gunicorn --preload, multiprocessing's fork, Celery prefork) exports its own spans."""
import os
import threading
import time
import unittest
import warnings

from tests.helpers import fresh, reset_context


@unittest.skipUnless(hasattr(os, "fork") and hasattr(os, "register_at_fork"), "needs os.fork")
class TestForkedChild(unittest.TestCase):
    def setUp(self):
        reset_context()
        self.captured = []
        self.client = fresh(self.captured)

    def _in_child(self, body):
        """Run body() in a forked child; returns the bytes it wrote to the pipe."""
        read_fd, write_fd = os.pipe()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pid = os.fork()
        if pid == 0:
            code = 1
            try:
                os.close(read_fd)
                os.write(write_fd, body().encode())
                code = 0
            except BaseException as exc:
                try:
                    os.write(write_fd, f"child failed: {exc!r}".encode())
                except Exception:
                    pass
            finally:
                os._exit(code)
        os.close(write_fd)
        chunks = []
        while True:
            chunk = os.read(read_fd, 4096)
            if not chunk:
                break
            chunks.append(chunk)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        out = b"".join(chunks).decode()
        self.assertEqual(os.waitstatus_to_exitcode(status), 0, out)
        return out

    def test_the_child_exports_from_its_own_thread_while_the_parent_keeps_going(self):
        parent_thread = self.client._thread
        client = self.client

        def body():
            thread = client._thread
            if thread is parent_thread or not thread.is_alive():
                return "no export thread in the child"
            end, _ = client.start_span("in-child")
            end("OK")
            client._flush_now.set()
            deadline = time.time() + 5
            while time.time() < deadline:
                if any(e.operation == "in-child" for e in self.captured):
                    return "exported by the child's thread"
                time.sleep(0.02)
            return "queued but never exported"

        self.assertEqual(self._in_child(body), "exported by the child's thread")
        self.assertIs(self.client._thread, parent_thread, "the parent keeps its own export thread")
        end, _ = self.client.start_span("in-parent")
        end("OK")
        self.client.flush()
        self.assertTrue(any(e.operation == "in-parent" for e in self.captured))
        self.assertFalse(any(e.operation == "in-child" for e in self.captured),
                         "a span the child queued is the child's to export")

    def test_a_lock_held_by_another_thread_at_the_fork_does_not_block_the_child(self):
        held, release = threading.Event(), threading.Event()

        def holder():
            with self.client._lock:
                held.set()
                release.wait(10)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        held.wait(5)
        client = self.client
        try:
            def body():
                if not client._lock.acquire(timeout=2):
                    return "child deadlocked on a lock copied in the held state"
                client._lock.release()
                end, _ = client.start_span("child-after-held-lock")
                end("OK")
                return "ok"

            self.assertEqual(self._in_child(body), "ok")
        finally:
            release.set()
            t.join(5)

    def test_the_queue_the_parent_had_is_not_sent_twice(self):
        end, _ = self.client.start_span("queued-before-fork")
        end("OK")
        client = self.client

        def body():
            return str(len(client._queue))

        self.assertEqual(self._in_child(body), "0")
        self.client.flush()
        self.assertEqual(len([e for e in self.captured if e.operation == "queued-before-fork"]), 1)


if __name__ == "__main__":
    unittest.main()
