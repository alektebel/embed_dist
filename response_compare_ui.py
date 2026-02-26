import tkinter as tk
from tkinter import ttk
import threading

import numpy as np

from validator import load_model, load_classifier, is_refusal


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def classify(sim_ab: float, is_ref_a: bool, is_ref_b: bool) -> tuple[float, str]:
    if is_ref_a and is_ref_b:
        return 1.0, "dual_refusal"
    if is_ref_a ^ is_ref_b:
        return -1.0, "guardrail_conflict"
    if sim_ab >= 0.80:
        return sim_ab, "agreement"
    if sim_ab >= 0.40:
        return sim_ab, "partial_divergence"
    return sim_ab, "divergence"


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("LLM Response Comparator")
        root.geometry("980x700")

        self.status = ttk.Label(root, text="Loading model and classifier...")
        self.status.pack(anchor="w", padx=12, pady=(10, 6))

        self.input_a = tk.Text(root, height=10, wrap="word")
        self.input_b = tk.Text(root, height=10, wrap="word")

        ttk.Label(root, text="LLM Response A").pack(anchor="w", padx=12)
        self.input_a.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        ttk.Label(root, text="LLM Response B").pack(anchor="w", padx=12)
        self.input_b.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        self.btn = ttk.Button(root, text="Compare", command=self.on_compare, state="disabled")
        self.btn.pack(anchor="w", padx=12, pady=(0, 8))

        self.output = tk.Text(root, height=12, wrap="word", state="disabled")
        ttk.Label(root, text="Result").pack(anchor="w", padx=12)
        self.output.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.model = None
        self.classifier = None
        self.threshold = 0.5
        self._loading_done = False
        self._loading_error = None
        self._compare_done = False
        self._compare_error = None
        self._compare_result = None
        self._show_loading_screen()
        self._start_loading()

    def _show_loading_screen(self):
        self.loading = tk.Toplevel(self.root)
        self.loading.title("Loading")
        self.loading.geometry("420x140")
        self.loading.resizable(False, False)
        self.loading.transient(self.root)
        self.loading.grab_set()
        self.loading.protocol("WM_DELETE_WINDOW", lambda: None)

        ttk.Label(
            self.loading,
            text="Loading model weights. This can take a while...",
        ).pack(anchor="w", padx=14, pady=(14, 6))
        ttk.Label(
            self.loading,
            text="The UI will unlock automatically when ready.",
        ).pack(anchor="w", padx=14, pady=(0, 8))

        self.pb = ttk.Progressbar(self.loading, mode="indeterminate", length=380)
        self.pb.pack(anchor="w", padx=14, pady=(0, 10))
        self.pb.start(12)

    def _start_loading(self):
        thread = threading.Thread(target=self._load_assets_worker, daemon=True)
        thread.start()
        self.root.after(120, self._poll_loading)

    def _load_assets_worker(self):
        try:
            self.model = load_model()
            self.classifier, self.threshold = load_classifier()
        except Exception as exc:
            self._loading_error = exc
        finally:
            self._loading_done = True

    def _poll_loading(self):
        if not self._loading_done:
            self.root.after(120, self._poll_loading)
            return
        self.pb.stop()
        self.loading.grab_release()
        self.loading.destroy()
        if self._loading_error is not None:
            self.status.config(text=f"Load failed: {self._loading_error}")
            self._set_output(f"Failed to load assets:\n{self._loading_error}")
            return
        self.status.config(text="Ready.")
        self.btn.config(state="normal")

    def _set_output(self, text: str):
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert("1.0", text)
        self.output.config(state="disabled")

    def _show_busy_screen(self, title: str, line1: str, line2: str):
        self.busy = tk.Toplevel(self.root)
        self.busy.title(title)
        self.busy.geometry("420x140")
        self.busy.resizable(False, False)
        self.busy.transient(self.root)
        self.busy.grab_set()
        self.busy.protocol("WM_DELETE_WINDOW", lambda: None)

        ttk.Label(self.busy, text=line1).pack(anchor="w", padx=14, pady=(14, 6))
        ttk.Label(self.busy, text=line2).pack(anchor="w", padx=14, pady=(0, 8))
        self.busy_pb = ttk.Progressbar(self.busy, mode="indeterminate", length=380)
        self.busy_pb.pack(anchor="w", padx=14, pady=(0, 10))
        self.busy_pb.start(12)

    def on_compare(self):
        text_a = self.input_a.get("1.0", tk.END).strip()
        text_b = self.input_b.get("1.0", tk.END).strip()

        if not text_a or not text_b:
            self._set_output("Please fill both response boxes.")
            return

        self.status.config(text="Computing embeddings...")
        self.btn.config(state="disabled")
        self._compare_done = False
        self._compare_error = None
        self._compare_result = None
        self._show_busy_screen(
            "Comparing",
            "Computing embeddings and classification...",
            "The UI will unlock automatically when done.",
        )
        thread = threading.Thread(
            target=self._compare_worker, args=(text_a, text_b), daemon=True
        )
        thread.start()
        self.root.after(120, self._poll_compare)

    def _compare_worker(self, text_a: str, text_b: str):
        try:
            emb_a, emb_b = self.model.encode([text_a, text_b], convert_to_numpy=True)
            sim_ab = cosine_similarity(emb_a, emb_b)
            dist_ab = 1.0 - sim_ab
            ref_a, conf_a = is_refusal(self.classifier, emb_a, self.threshold)
            ref_b, conf_b = is_refusal(self.classifier, emb_b, self.threshold)
            composite, label = classify(sim_ab, ref_a, ref_b)
            self._compare_result = (
                f"Cosine similarity : {sim_ab:+.6f}\n"
                f"Embedding distance: {dist_ab:.6f}\n"
                f"Probable class    : {label}\n"
                f"Composite score   : {composite:+.6f}\n\n"
                f"Response A refusal: {'Yes' if ref_a else 'No'} (confidence: {conf_a:.3f})\n"
                f"Response B refusal: {'Yes' if ref_b else 'No'} (confidence: {conf_b:.3f})\n"
            )
        except Exception as exc:
            self._compare_error = exc
        finally:
            self._compare_done = True

    def _poll_compare(self):
        if not self._compare_done:
            self.root.after(120, self._poll_compare)
            return
        self.busy_pb.stop()
        self.busy.grab_release()
        self.busy.destroy()
        if self._compare_error is not None:
            self._set_output(f"Comparison failed:\n{self._compare_error}")
            self.status.config(text="Comparison failed.")
        else:
            self._set_output(self._compare_result)
            self.status.config(text="Done.")
        self.btn.config(state="normal")


def main():
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
