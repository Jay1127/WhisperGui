import json
import os
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import tqdm


_progress_local = threading.local()
_progress_hooked = False


class _WhisperProgressBar(tqdm.tqdm):
    def update(self, n=1):
        super().update(n)
        callback = getattr(_progress_local, "callback", None)
        if callback and self.total:
            callback(self.n, self.total)


class whisper_progress:
    def __init__(self, callback):
        self.callback = callback
        self.previous_callback = None

    def __enter__(self):
        _install_progress_hook()
        self.previous_callback = getattr(_progress_local, "callback", None)
        _progress_local.callback = self.callback
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        _progress_local.callback = self.previous_callback


def _install_progress_hook():
    global _progress_hooked
    if _progress_hooked:
        return
    import whisper.transcribe

    transcribe_module = sys.modules["whisper.transcribe"]
    transcribe_module.tqdm.tqdm = _WhisperProgressBar
    _progress_hooked = True


class WhisperGui(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Whisper GUI")
        self.geometry("1120x860")
        self.minsize(940, 660)

        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.engine = tk.StringVar(value="openai-whisper")
        self.task = tk.StringVar(value="transcribe")
        self.model_name = tk.StringVar(value="small")
        self.language = tk.StringVar()
        self.output_format = tk.StringVar(value="txt")

        self.device = tk.StringVar(value="auto")
        self.model_dir = tk.StringVar()
        self.initial_prompt = tk.StringVar()
        self.condition_on_previous_text = tk.BooleanVar(value=True)
        self.carry_initial_prompt = tk.BooleanVar(value=False)
        self.word_timestamps = tk.BooleanVar(value=False)
        self.highlight_words = tk.BooleanVar(value=False)
        self.fp16 = tk.BooleanVar(value=True)
        self.verbose = tk.BooleanVar(value=False)

        self.temperature = tk.StringVar()
        self.temperature_increment_on_fallback = tk.StringVar()
        self.best_of = tk.StringVar()
        self.beam_size = tk.StringVar()
        self.patience = tk.StringVar()
        self.length_penalty = tk.StringVar()
        self.suppress_tokens = tk.StringVar()
        self.threads = tk.StringVar()
        self.clip_timestamps = tk.StringVar()
        self.max_line_width = tk.StringVar()
        self.max_line_count = tk.StringVar()
        self.max_words_per_line = tk.StringVar()
        self.prepend_punctuations = tk.StringVar(value="\"'\u201c\u00bf([{-")
        self.append_punctuations = tk.StringVar(value="\"'.\u3002,\uff0c!\uff01?\uff1f:\uff1a\u201d)]}\u3001")
        self.hallucination_silence_threshold = tk.StringVar()
        self.compression_ratio_threshold = tk.StringVar()
        self.logprob_threshold = tk.StringVar()
        self.no_speech_threshold = tk.StringVar()
        self.compute_type = tk.StringVar(value="default")
        self.vad_filter = tk.BooleanVar(value=False)
        self.hotwords = tk.StringVar()
        self.batch_size = tk.StringVar()

        self.status_text = tk.StringVar(value="Ready")
        self.progress_text = tk.StringVar(value="0%")
        self.progress_value = tk.DoubleVar(value=0)
        self.worker_thread = None

        self.model_values = ["tiny", "base", "small", "medium", "large", "turbo"]
        self.engine_values = ["openai-whisper", "faster-whisper"]
        self.compute_type_values = ["default", "float32", "float16", "int8"]
        self.language_values = ["", "Korean", "English", "Japanese", "Chinese", "Spanish", "French", "German"]
        self.output_formats = ["txt", "srt", "vtt", "json", "tsv", "all"]

        self._build_ui()
        self.engine.trace_add("write", self._sync_engine_options)
        self.highlight_words.trace_add("write", self._sync_word_timestamps)
        self.word_timestamps.trace_add("write", self._sync_word_timestamp_options)
        self._sync_engine_options()
        self._sync_word_timestamp_options()
        self._update_dependency_status()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Input audio").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(top, textvariable=self.input_path).grid(row=0, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Browse", command=self.choose_input).grid(row=0, column=2, padx=(8, 0), pady=4)

        ttk.Label(top, text="Output file").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(top, textvariable=self.output_path).grid(row=1, column=1, sticky="ew", pady=4)
        ttk.Button(top, text="Browse", command=self.choose_output).grid(row=1, column=2, padx=(8, 0), pady=4)

        options = ttk.Frame(top)
        options.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        options.columnconfigure(0, weight=1)
        options.columnconfigure(1, weight=1)
        options.columnconfigure(2, weight=1)

        self._build_basic_options(options).grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._build_decoding_options(options).grid(row=0, column=1, sticky="nsew", padx=6)
        self.timestamp_frame = self._build_timestamp_subtitle_options(options)
        self.timestamp_frame.grid(row=0, column=2, sticky="nsew", padx=(6, 0))
        self._build_runtime_options(options).grid(row=1, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        self._build_faster_options(options).grid(row=2, column=0, columnspan=3, sticky="ew", pady=(8, 0))

        actions = ttk.Frame(top)
        actions.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        actions.columnconfigure(4, weight=1)
        self.convert_button = ttk.Button(actions, text="Convert", command=self.start_convert)
        self.convert_button.grid(row=0, column=0, padx=(0, 8))
        ttk.Button(actions, text="Open output", command=self.open_output).grid(row=0, column=1, padx=(0, 8))
        ttk.Button(actions, text="Open folder", command=self.open_output_folder).grid(row=0, column=2, padx=(0, 8))
        ttk.Button(actions, text="Clear preview", command=self.clear_preview).grid(row=0, column=3, padx=(0, 8))

        progress_frame = ttk.Frame(top)
        progress_frame.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        progress_frame.columnconfigure(0, weight=1)
        self.progress = ttk.Progressbar(progress_frame, variable=self.progress_value, maximum=100)
        self.progress.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ttk.Label(progress_frame, textvariable=self.progress_text, width=8, anchor="e").grid(row=0, column=1, sticky="e")

        body = ttk.PanedWindow(self, orient=tk.VERTICAL)
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

        preview_frame = ttk.LabelFrame(body, text="Text preview", padding=8)
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        self.preview = tk.Text(preview_frame, wrap="word", undo=False)
        preview_scroll = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview.yview)
        self.preview.configure(yscrollcommand=preview_scroll.set)
        self.preview.grid(row=0, column=0, sticky="nsew")
        preview_scroll.grid(row=0, column=1, sticky="ns")

        log_frame = ttk.LabelFrame(body, text="Log", padding=8)
        log_frame.rowconfigure(0, weight=1)
        log_frame.columnconfigure(0, weight=1)
        self.log = tk.Text(log_frame, height=8, wrap="word", state="disabled")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky="nsew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        body.add(preview_frame, weight=4)
        body.add(log_frame, weight=1)

        ttk.Label(self, textvariable=self.status_text, anchor="w", padding=(12, 0, 12, 8)).grid(
            row=2, column=0, sticky="ew"
        )

    def _build_basic_options(self, parent):
        frame = ttk.LabelFrame(parent, text="Whisper options", padding=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        ttk.Label(frame, text="Engine").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(frame, textvariable=self.engine, values=self.engine_values, width=16, state="readonly").grid(
            row=0, column=1, sticky="ew", pady=3
        )
        ttk.Label(frame, text="Model").grid(row=0, column=2, sticky="w", padx=(12, 8), pady=3)
        ttk.Combobox(frame, textvariable=self.model_name, values=self.model_values, width=16, state="readonly").grid(
            row=0, column=3, sticky="ew", pady=3
        )
        ttk.Label(frame, text="Task").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(frame, textvariable=self.task, values=["transcribe", "translate"], width=16, state="readonly").grid(
            row=1, column=1, sticky="ew", pady=3
        )
        ttk.Label(frame, text="Output format").grid(row=1, column=2, sticky="w", padx=(12, 8), pady=3)
        ttk.Combobox(
            frame,
            textvariable=self.output_format,
            values=self.output_formats,
            width=16,
            state="readonly",
        ).grid(row=1, column=3, sticky="ew", pady=3)

        ttk.Label(frame, text="Language").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Combobox(frame, textvariable=self.language, values=self.language_values).grid(
            row=2, column=1, sticky="ew", pady=3
        )
        ttk.Label(frame, text="Device").grid(row=2, column=2, sticky="w", padx=(12, 8), pady=3)
        ttk.Combobox(frame, textvariable=self.device, values=["auto", "cpu", "cuda"], width=16, state="readonly").grid(
            row=2, column=3, sticky="ew", pady=3
        )
        ttk.Label(frame, text="Model dir").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=3)
        model_dir_row = ttk.Frame(frame)
        model_dir_row.grid(row=3, column=1, columnspan=3, sticky="ew", pady=3)
        model_dir_row.columnconfigure(0, weight=1)
        ttk.Entry(model_dir_row, textvariable=self.model_dir).grid(row=0, column=0, sticky="ew")
        ttk.Button(model_dir_row, text="...", width=3, command=self.choose_model_dir).grid(row=0, column=1, padx=(4, 0))

        ttk.Label(frame, text="Initial prompt").grid(row=4, column=0, sticky="w", padx=(0, 8), pady=3)
        ttk.Entry(frame, textvariable=self.initial_prompt).grid(row=4, column=1, columnspan=3, sticky="ew", pady=3)

        checks = ttk.Frame(frame)
        checks.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 2))
        checks.columnconfigure(0, weight=1)
        checks.columnconfigure(1, weight=1)
        ttk.Checkbutton(checks, text="Condition on previous text", variable=self.condition_on_previous_text).grid(
            row=0, column=0, sticky="w", padx=(0, 16)
        )
        ttk.Checkbutton(checks, text="Word timestamps", variable=self.word_timestamps).grid(
            row=0, column=1, sticky="w", padx=(0, 16)
        )
        ttk.Checkbutton(checks, text="Carry initial prompt", variable=self.carry_initial_prompt).grid(
            row=1, column=0, sticky="w", padx=(0, 16)
        )
        self.highlight_words_check = ttk.Checkbutton(checks, text="Highlight words", variable=self.highlight_words)
        self.highlight_words_check.grid(row=1, column=1, sticky="w", padx=(0, 16))
        ttk.Checkbutton(checks, text="fp16", variable=self.fp16).grid(row=2, column=0, sticky="w", padx=(0, 16))
        ttk.Checkbutton(checks, text="Verbose", variable=self.verbose).grid(row=2, column=1, sticky="w")

        return frame

    def _build_decoding_options(self, parent):
        frame = ttk.LabelFrame(parent, text="Decoding options", padding=8)
        for col in [1, 3]:
            frame.columnconfigure(col, weight=1)

        fields = [
            ("Temperature", self.temperature),
            ("Temp fallback", self.temperature_increment_on_fallback),
            ("Best of", self.best_of),
            ("Beam size", self.beam_size),
            ("Patience", self.patience),
            ("Length penalty", self.length_penalty),
            ("Suppress tokens", self.suppress_tokens),
            ("Compression threshold", self.compression_ratio_threshold),
            ("Logprob threshold", self.logprob_threshold),
            ("No speech threshold", self.no_speech_threshold),
        ]
        for index, (label, variable) in enumerate(fields):
            row = index // 2
            label_col = (index % 2) * 2
            left_pad = 0 if label_col == 0 else 18
            right_pad = 18 if label_col == 0 else 0
            ttk.Label(frame, text=label).grid(row=row, column=label_col, sticky="w", padx=(left_pad, 8), pady=3)
            ttk.Entry(frame, textvariable=variable, width=18).grid(
                row=row,
                column=label_col + 1,
                sticky="ew",
                padx=(0, right_pad),
                pady=3,
            )

        return frame

    def _build_timestamp_subtitle_options(self, parent):
        frame = ttk.LabelFrame(parent, text="Timestamp / subtitle options", padding=8)
        frame.columnconfigure(1, weight=1)

        self.word_timestamp_widgets = []
        word_timestamp_fields = [
            ("Max line width", self.max_line_width),
            ("Max line count", self.max_line_count),
            ("Max words per line", self.max_words_per_line),
            ("Prepend punct.", self.prepend_punctuations),
            ("Append punct.", self.append_punctuations),
        ]
        for index, (label, variable) in enumerate(word_timestamp_fields):
            label_widget = ttk.Label(frame, text=label)
            label_widget.grid(row=index, column=0, sticky="w", padx=(0, 8), pady=3)
            entry = ttk.Entry(frame, textvariable=variable)
            entry.grid(row=index, column=1, sticky="ew", pady=3)
            self.word_timestamp_widgets.extend([label_widget, entry])

        return frame

    def _build_runtime_options(self, parent):
        frame = ttk.LabelFrame(parent, text="Runtime options", padding=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        runtime_fields = [
            ("Threads", self.threads),
            ("Clip timestamps", self.clip_timestamps),
        ]
        for index, (label, variable) in enumerate(runtime_fields):
            label_col = index * 2
            left_pad = 0 if label_col == 0 else 18
            right_pad = 18 if label_col == 0 else 0
            ttk.Label(frame, text=label).grid(row=0, column=label_col, sticky="w", padx=(left_pad, 8), pady=3)
            ttk.Entry(frame, textvariable=variable).grid(
                row=0,
                column=label_col + 1,
                sticky="ew",
                padx=(0, right_pad),
                pady=3,
            )

        self.hallucination_widgets = []
        label_widget = ttk.Label(frame, text="Silence threshold")
        label_widget.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = ttk.Entry(frame, textvariable=self.hallucination_silence_threshold)
        entry.grid(row=1, column=1, sticky="ew", padx=(0, 18), pady=3)
        self.hallucination_widgets.extend([label_widget, entry])

        return frame

    def _build_faster_options(self, parent):
        frame = ttk.LabelFrame(parent, text="Faster Whisper options", padding=8)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(3, weight=1)

        self.faster_widgets = []

        label_widget = ttk.Label(frame, text="Compute type")
        label_widget.grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        compute_combo = ttk.Combobox(
            frame,
            textvariable=self.compute_type,
            values=self.compute_type_values,
            width=16,
            state="readonly",
        )
        compute_combo.grid(row=0, column=1, sticky="ew", padx=(0, 18), pady=3)
        self.faster_widgets.extend([label_widget, compute_combo])

        vad_check = ttk.Checkbutton(frame, text="VAD filter", variable=self.vad_filter)
        vad_check.grid(row=0, column=2, sticky="w", padx=(18, 8), pady=3)
        self.faster_widgets.append(vad_check)

        label_widget = ttk.Label(frame, text="Batch size")
        label_widget.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        entry = ttk.Entry(frame, textvariable=self.batch_size)
        entry.grid(row=1, column=1, sticky="ew", padx=(0, 18), pady=3)
        self.faster_widgets.extend([label_widget, entry])

        label_widget = ttk.Label(frame, text="Hotwords")
        label_widget.grid(row=1, column=2, sticky="w", padx=(18, 8), pady=3)
        entry = ttk.Entry(frame, textvariable=self.hotwords)
        entry.grid(row=1, column=3, sticky="ew", pady=3)
        self.faster_widgets.extend([label_widget, entry])

        return frame

    def choose_input(self):
        path = filedialog.askopenfilename(
            title="Choose audio file",
            filetypes=[
                ("Audio files", "*.mp3 *.mp4 *.mpeg *.mpga *.m4a *.wav *.webm *.flac *.ogg"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return
        self.input_path.set(path)
        if not self.output_path.get().strip():
            self.output_path.set(str(Path(path).with_suffix(".txt")))

    def choose_output(self):
        path = filedialog.asksaveasfilename(
            title="Choose output file",
            defaultextension=".txt",
            filetypes=[
                ("Text", "*.txt"),
                ("Subtitle", "*.srt *.vtt"),
                ("JSON", "*.json"),
                ("TSV", "*.tsv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.output_path.set(path)

    def choose_model_dir(self):
        path = filedialog.askdirectory(title="Choose Whisper model directory")
        if path:
            self.model_dir.set(path)

    def start_convert(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showwarning("Busy", "Conversion is already running.")
            return

        input_path = self.input_path.get().strip()
        output_path = self.output_path.get().strip()

        if not input_path:
            messagebox.showwarning("Input required", "Choose an audio file first.")
            return
        if not os.path.exists(input_path):
            messagebox.showerror("Input not found", input_path)
            return
        if not output_path:
            messagebox.showwarning("Output required", "Choose an output file.")
            return

        self._normalize_related_options()
        try:
            options = self._collect_options()
        except ValueError as exc:
            messagebox.showerror("Invalid option", str(exc))
            return

        self._set_busy(True)
        self._set_progress(0, 1)
        self.status_text.set("Converting...")
        self.append_log(f"Input: {input_path}")
        self.append_log(f"Engine: {options['engine']}")

        self.worker_thread = threading.Thread(
            target=self._run_conversion,
            args=(input_path, output_path, options),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_conversion(self, input_path, output_path, options):
        try:
            text, saved_path = self._convert_with_whisper(input_path, output_path, options)
            self.after(0, self._show_result, text, saved_path)
        except Exception as exc:
            self.after(0, self._show_error, exc)
        finally:
            self.after(0, self._set_busy, False)

    def _convert_with_whisper(self, input_path, output_path, options):
        if options["engine"] == "faster-whisper":
            return self._convert_with_faster_whisper(input_path, output_path, options)
        return self._convert_with_openai_whisper(input_path, output_path, options)

    def _convert_with_openai_whisper(self, input_path, output_path, options):
        try:
            import torch
            import whisper
        except ImportError as exc:
            raise RuntimeError("Install Whisper first: pip install openai-whisper") from exc

        threads = options["threads"]
        if threads is not None:
            torch.set_num_threads(threads)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        device = None if options["device"] == "auto" else options["device"]
        self.after(0, self.append_log, f"Loading model: {options['model_name']}")
        model = whisper.load_model(options["model_name"], device=device, download_root=options["model_dir"])

        self.after(0, self.append_log, "Transcribing...")
        with whisper_progress(self._on_whisper_progress):
            result = model.transcribe(input_path, **options["transcribe_options"])

        return self._write_result(result, input_path, output_path, options)

    def _convert_with_faster_whisper(self, input_path, output_path, options):
        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise RuntimeError("Install faster-whisper first: pip install faster-whisper") from exc

        model_options = options["faster_model_options"]
        transcribe_options = options["faster_transcribe_options"]
        batch_size = options["batch_size"]

        self.after(0, self.append_log, f"Loading faster-whisper model: {options['model_name']}")
        model = WhisperModel(options["model_name"], **model_options)
        transcriber = BatchedInferencePipeline(model=model) if batch_size else model

        if batch_size:
            transcribe_options["batch_size"] = batch_size

        self.after(0, self.append_log, "Transcribing with faster-whisper...")
        segments, info = transcriber.transcribe(input_path, **transcribe_options)
        duration = getattr(info, "duration", None)
        segment_dicts = []
        text_parts = []

        for segment in segments:
            segment_dict = {
                "id": len(segment_dicts),
                "seek": 0,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "tokens": [],
                "temperature": 0,
                "avg_logprob": 0,
                "compression_ratio": 0,
                "no_speech_prob": 0,
            }
            words = getattr(segment, "words", None)
            if words:
                segment_dict["words"] = [
                    {
                        "word": word.word,
                        "start": word.start,
                        "end": word.end,
                        "probability": word.probability,
                    }
                    for word in words
                ]
            segment_dicts.append(segment_dict)
            text_parts.append(segment.text)
            if duration:
                self._on_whisper_progress(segment.end, duration)

        if duration:
            self._on_whisper_progress(duration, duration)
        else:
            self._on_whisper_progress(1, 1)

        result = {
            "text": "".join(text_parts).strip(),
            "segments": segment_dicts,
            "language": getattr(info, "language", None),
        }
        return self._write_result(result, input_path, output_path, options)

    def _write_result(self, result, input_path, output_path, options):
        from whisper.utils import get_writer

        output_format = options["output_format"]
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        writer = get_writer(output_format, str(output.parent))
        writer(result, input_path, options["writer_options"])

        generated = self._find_generated_output(input_path, output.parent, output_format)
        if generated is not None and generated.resolve() != output.resolve() and output_format != "all":
            output.write_text(generated.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")

        if output_format == "all":
            txt_path = output.parent / f"{Path(input_path).stem}.txt"
            if txt_path.exists():
                return txt_path.read_text(encoding="utf-8", errors="replace"), str(output.parent)
            return result.get("text", ""), str(output.parent)

        if output.exists():
            return output.read_text(encoding="utf-8", errors="replace"), str(output)

        return result.get("text", ""), str(output.parent)

    def _on_whisper_progress(self, current, total):
        self.after(0, self._set_progress, current, total)

    def _set_progress(self, current, total):
        if not total:
            percent = 0
        else:
            percent = max(0, min(100, current / total * 100))
        self.progress_value.set(percent)
        self.progress_text.set(f"{percent:.0f}%")

    def _collect_options(self):
        engine = self.engine.get()
        options = {
            "engine": engine,
            "model_name": self.model_name.get(),
            "device": self.device.get(),
            "model_dir": self.model_dir.get().strip() or None,
            "output_format": self.output_format.get(),
            "threads": self._optional_int_value(self.threads.get(), "Threads"),
            "writer_options": self._writer_options(),
        }
        if engine == "faster-whisper":
            options["faster_model_options"] = self._faster_model_options()
            options["faster_transcribe_options"] = self._faster_transcribe_options()
            options["batch_size"] = self._positive_optional_int_value(self.batch_size.get(), "Batch size")
        else:
            options["transcribe_options"] = self._transcribe_options()
        return options

    def _transcribe_options(self):
        options = {
            "task": self.task.get(),
            "verbose": self.verbose.get(),
            "condition_on_previous_text": self.condition_on_previous_text.get(),
            "word_timestamps": self.word_timestamps.get(),
            "fp16": self.fp16.get(),
        }
        self._set_option(options, "language", self.language.get())
        self._set_option(options, "initial_prompt", self.initial_prompt.get())
        self._set_option(options, "temperature", self._temperature_value())
        self._set_option(options, "best_of", self._optional_int_value(self.best_of.get(), "Best of"))
        self._set_option(options, "beam_size", self._optional_int_value(self.beam_size.get(), "Beam size"))
        self._set_option(options, "patience", self._optional_float_value(self.patience.get(), "Patience"))
        self._set_option(options, "length_penalty", self._optional_float_value(self.length_penalty.get(), "Length penalty"))
        self._set_option(options, "suppress_tokens", self.suppress_tokens.get())
        self._set_option(options, "clip_timestamps", self.clip_timestamps.get())
        options["prepend_punctuations"] = self.prepend_punctuations.get()
        options["append_punctuations"] = self.append_punctuations.get()
        self._set_option(
            options,
            "hallucination_silence_threshold",
            self._optional_float_value(self.hallucination_silence_threshold.get(), "Silence threshold"),
        )
        self._set_option(
            options,
            "compression_ratio_threshold",
            self._optional_float_value(self.compression_ratio_threshold.get(), "Compression threshold"),
        )
        self._set_option(options, "logprob_threshold", self._optional_float_value(self.logprob_threshold.get(), "Logprob threshold"))
        self._set_option(options, "no_speech_threshold", self._optional_float_value(self.no_speech_threshold.get(), "No speech threshold"))

        if self.carry_initial_prompt.get():
            options["carry_initial_prompt"] = True
        return options

    def _faster_model_options(self):
        options = {
            "device": self.device.get(),
        }
        self._set_option(options, "download_root", self.model_dir.get())
        if self.threads.get().strip():
            options["cpu_threads"] = self._optional_int_value(self.threads.get(), "Threads")
        if self.compute_type.get() != "default":
            options["compute_type"] = self.compute_type.get()
        return options

    def _faster_transcribe_options(self):
        options = {
            "task": self.task.get(),
            "condition_on_previous_text": self.condition_on_previous_text.get(),
            "word_timestamps": self.word_timestamps.get(),
        }
        if self.vad_filter.get():
            options["vad_filter"] = True
        self._set_option(options, "language", self._faster_language(self.language.get()))
        self._set_option(options, "initial_prompt", self.initial_prompt.get())
        self._set_option(options, "temperature", self._temperature_value())
        self._set_option(options, "best_of", self._optional_int_value(self.best_of.get(), "Best of"))
        self._set_option(options, "beam_size", self._optional_int_value(self.beam_size.get(), "Beam size"))
        self._set_option(options, "patience", self._optional_float_value(self.patience.get(), "Patience"))
        self._set_option(options, "length_penalty", self._optional_float_value(self.length_penalty.get(), "Length penalty"))
        self._set_option(options, "suppress_tokens", self._faster_suppress_tokens(self.suppress_tokens.get()))
        self._set_option(options, "clip_timestamps", self.clip_timestamps.get())
        self._set_option(options, "hotwords", self.hotwords.get())
        options["prepend_punctuations"] = self.prepend_punctuations.get()
        options["append_punctuations"] = self.append_punctuations.get()
        self._set_option(
            options,
            "compression_ratio_threshold",
            self._optional_float_value(self.compression_ratio_threshold.get(), "Compression threshold"),
        )
        self._set_option(options, "log_prob_threshold", self._optional_float_value(self.logprob_threshold.get(), "Logprob threshold"))
        self._set_option(options, "no_speech_threshold", self._optional_float_value(self.no_speech_threshold.get(), "No speech threshold"))
        return options

    def _faster_language(self, value):
        value = value.strip()
        language_map = {
            "Korean": "ko",
            "English": "en",
            "Japanese": "ja",
            "Chinese": "zh",
            "Spanish": "es",
            "French": "fr",
            "German": "de",
        }
        return language_map.get(value, value)

    def _faster_suppress_tokens(self, value):
        value = value.strip()
        if not value:
            return None
        try:
            return [int(token.strip()) for token in value.split(",") if token.strip()]
        except ValueError as exc:
            raise ValueError("Suppress tokens must be comma-separated token numbers for faster-whisper.") from exc

    def _writer_options(self):
        return {
            "highlight_words": self.highlight_words.get(),
            "max_line_width": self._optional_int_value(self.max_line_width.get(), "Max line width"),
            "max_line_count": self._optional_int_value(self.max_line_count.get(), "Max line count"),
            "max_words_per_line": self._optional_int_value(self.max_words_per_line.get(), "Max words per line"),
        }

    def _temperature_value(self):
        base = self.temperature.get().strip()
        increment = self.temperature_increment_on_fallback.get().strip()
        if not base and not increment:
            return None
        if not base:
            base = "0"
        if not increment:
            return self._float_value(base, "Temperature")

        start = self._float_value(base, "Temperature")
        step = self._float_value(increment, "Temp fallback")
        if step <= 0:
            raise ValueError("Temp fallback must be greater than 0.")
        values = []
        current = start
        while current <= 1.0:
            values.append(round(current, 2))
            current += step
        return tuple(values) if values else start

    def _find_generated_output(self, input_path, output_dir, output_format):
        if output_format == "all":
            return None
        generated = output_dir / f"{Path(input_path).stem}.{output_format}"
        return generated if generated.exists() else None

    def _show_result(self, text, output_path):
        self.preview.delete("1.0", tk.END)
        self.preview.insert(tk.END, text)
        self.status_text.set(f"Saved: {output_path}")
        self.append_log(f"Saved: {output_path}")

    def _show_error(self, exc):
        self.status_text.set("Conversion failed")
        self.append_log(f"ERROR: {exc}")
        messagebox.showerror("Conversion failed", str(exc))

    def open_output(self):
        output_path = self.output_path.get().strip()
        if not output_path or not os.path.exists(output_path):
            messagebox.showwarning("Output not found", "No output file exists yet.")
            return
        os.startfile(output_path)

    def open_output_folder(self):
        output_path = self.output_path.get().strip()
        if not output_path:
            messagebox.showwarning("Output required", "Choose an output file first.")
            return
        folder = Path(output_path).parent
        if not folder.exists():
            messagebox.showwarning("Folder not found", str(folder))
            return
        os.startfile(folder)

    def clear_preview(self):
        self.preview.delete("1.0", tk.END)

    def append_log(self, message):
        self.log.configure(state="normal")
        self.log.insert(tk.END, str(message) + "\n")
        self.log.see(tk.END)
        self.log.configure(state="disabled")

    def _set_busy(self, busy):
        self.convert_button.configure(state="disabled" if busy else "normal")

    def _sync_engine_options(self, *_):
        if not hasattr(self, "faster_widgets"):
            return
        enabled = self.engine.get() == "faster-whisper"
        for child in self.faster_widgets:
            try:
                if child.winfo_class() == "TCombobox":
                    child.configure(state="readonly" if enabled else "disabled")
                else:
                    child.configure(state="normal" if enabled else "disabled")
            except tk.TclError:
                pass

    def _sync_word_timestamps(self, *_):
        if self.highlight_words.get() and not self.word_timestamps.get():
            self.word_timestamps.set(True)

    def _sync_word_timestamp_options(self, *_):
        if not hasattr(self, "word_timestamp_widgets"):
            return
        state = "normal" if self.word_timestamps.get() else "disabled"
        widgets = list(self.word_timestamp_widgets)
        if hasattr(self, "hallucination_widgets"):
            widgets.extend(self.hallucination_widgets)
        if hasattr(self, "highlight_words_check"):
            widgets.append(self.highlight_words_check)
        for child in widgets:
            try:
                child.configure(state=state)
            except tk.TclError:
                pass
        if not self.word_timestamps.get() and self.highlight_words.get():
            self.highlight_words.set(False)

    def _normalize_related_options(self):
        if self.highlight_words.get() and not self.word_timestamps.get():
            self.word_timestamps.set(True)
            self.append_log("Word timestamps was enabled because Highlight words requires it.")

    def _update_dependency_status(self):
        try:
            import whisper  # noqa: F401

            self.append_log("Whisper package found")
        except ImportError:
            self.append_log("Whisper package not found. Run: pip install openai-whisper")
        try:
            import faster_whisper  # noqa: F401

            self.append_log("faster-whisper package found")
        except ImportError:
            self.append_log("faster-whisper package not found. Run: pip install faster-whisper")

    def _set_option(self, options, key, value):
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        options[key] = value

    def _optional_int_value(self, value, label):
        value = str(value).strip()
        if not value:
            return None
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be an integer.") from exc

    def _positive_optional_int_value(self, value, label):
        value = self._optional_int_value(value, label)
        if value is not None and value <= 0:
            raise ValueError(f"{label} must be greater than 0.")
        return value

    def _optional_float_value(self, value, label):
        value = str(value).strip()
        if not value:
            return None
        return self._float_value(value, label)

    def _float_value(self, value, label):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError(f"{label} must be a number.") from exc


if __name__ == "__main__":
    app = WhisperGui()
    app.mainloop()
