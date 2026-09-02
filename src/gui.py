"""Minimal Tkinter GUI for reviewing colony segmentation quality.

This is a segmentation quality-control tool, not a general image editor:
open a folder of plate photographs, tune segmentation parameters per image
with the overlay updating live, optionally save per-image overrides for
outliers, and export a batch CSV once satisfied. No drawing tools, filters,
undo stack, or menus beyond File > Open Folder.
"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from src.batch import draw_overlay_on_axes, find_images, run_batch  # noqa: E402
from src.calibrate import CalibrationResult, calibrate  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.segment import load_image, segment_colony  # noqa: E402

DEFAULT_THRESHOLD_OFFSET = 0.0
DEFAULT_MIN_OBJECT_SIZE = 500
DEFAULT_DISH_DIAMETER_MM = 90.0


class ColonyReviewApp:
    """Main application window for reviewing and exporting colony segmentations."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Colony Biofilm Morphometrics -- Segmentation Review")

        self.input_dir: Path | None = None
        self.image_paths: list[Path] = []
        self.current_index: int = -1
        self.current_image = None
        # Per-image parameter overrides, keyed by filename: {"threshold_offset": ..., "min_object_size": ...}
        self.overrides: dict[str, dict] = {}

        self.threshold_offset_var = tk.DoubleVar(value=DEFAULT_THRESHOLD_OFFSET)
        self.min_object_size_var = tk.IntVar(value=DEFAULT_MIN_OBJECT_SIZE)
        self.dish_diameter_var = tk.DoubleVar(value=DEFAULT_DISH_DIAMETER_MM)

        self._build_menu()
        self._build_layout()
        self._set_controls_enabled(False)

    # ---- UI construction -------------------------------------------------

    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Folder...", command=self.open_folder)
        menubar.add_cascade(label="File", menu=file_menu)
        self.root.config(menu=menubar)

    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(5.5, 5.5))
        self.ax = self.figure.add_subplot(111)
        self.ax.axis("off")
        self.ax.text(0.5, 0.5, "File > Open Folder to begin", ha="center", va="center")
        self.canvas = FigureCanvasTkAgg(self.figure, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        nav = ttk.Frame(left)
        nav.pack(fill=tk.X, pady=4)
        self.prev_button = ttk.Button(nav, text="< Previous", command=self.previous_image)
        self.prev_button.pack(side=tk.LEFT)
        self.filename_label = ttk.Label(nav, text="No folder opened", anchor="center")
        self.filename_label.pack(side=tk.LEFT, expand=True, fill=tk.X)
        self.next_button = ttk.Button(nav, text="Next >", command=self.next_image)
        self.next_button.pack(side=tk.RIGHT)

        right = ttk.Frame(main, width=280)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0))

        ttk.Label(right, text="Segmentation parameters", font=("", 10, "bold")).pack(
            anchor="w", pady=(0, 4)
        )

        self.threshold_scale = tk.Scale(
            right,
            from_=-0.3,
            to=0.3,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="Threshold offset",
            variable=self.threshold_offset_var,
            command=self._on_param_change,
            length=240,
        )
        self.threshold_scale.pack(fill=tk.X)

        self.min_size_scale = tk.Scale(
            right,
            from_=0,
            to=5000,
            resolution=50,
            orient=tk.HORIZONTAL,
            label="Min object size (px)",
            variable=self.min_object_size_var,
            command=self._on_param_change,
            length=240,
        )
        self.min_size_scale.pack(fill=tk.X, pady=(6, 0))

        ttk.Label(right, text="Dish diameter (mm)").pack(anchor="w", pady=(10, 0))
        self.dish_entry = ttk.Entry(right, textvariable=self.dish_diameter_var, width=10)
        self.dish_entry.pack(anchor="w")
        self.dish_entry.bind("<Return>", self._on_param_change)
        self.dish_entry.bind("<FocusOut>", self._on_param_change)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        ttk.Label(right, text="Metrics", font=("", 10, "bold")).pack(anchor="w")
        self.metrics_text = tk.Text(
            right,
            height=12,
            width=32,
            state=tk.DISABLED,
            relief=tk.FLAT,
            background=self.root.cget("background"),
        )
        self.metrics_text.pack(anchor="w", pady=(2, 10))

        self.accept_button = ttk.Button(
            right,
            text="Accept and save parameters for this image",
            command=self.accept_parameters,
        )
        self.accept_button.pack(fill=tk.X, pady=(0, 6))

        self.batch_button = ttk.Button(
            right, text="Run batch with current settings", command=self.run_batch_export
        )
        self.batch_button.pack(fill=tk.X)

        self.status_label = ttk.Label(right, text="", foreground="gray", wraplength=260)
        self.status_label.pack(anchor="w", pady=(10, 0))

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for widget in (
            self.threshold_scale,
            self.min_size_scale,
            self.dish_entry,
            self.accept_button,
            self.batch_button,
        ):
            widget.configure(state=state)
        self._update_nav_buttons()

    def _update_nav_buttons(self) -> None:
        has_images = bool(self.image_paths)
        self.prev_button.configure(
            state=tk.NORMAL if has_images and self.current_index > 0 else tk.DISABLED
        )
        self.next_button.configure(
            state=tk.NORMAL
            if has_images and self.current_index < len(self.image_paths) - 1
            else tk.DISABLED
        )

    # ---- folder / navigation ----------------------------------------------

    def open_folder(self) -> None:
        folder = filedialog.askdirectory(title="Open folder of plate photographs")
        if not folder:
            return
        paths = find_images(Path(folder))
        if not paths:
            messagebox.showwarning(
                "No images found", f"No .jpg/.jpeg/.png/.tif files found in {folder}"
            )
            return

        self.input_dir = Path(folder)
        self.image_paths = paths
        self.current_index = 0
        self.overrides = {}
        self._set_controls_enabled(True)
        self._load_current_image()

    def next_image(self) -> None:
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self._load_current_image()

    def previous_image(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._load_current_image()

    def _load_current_image(self) -> None:
        path = self.image_paths[self.current_index]
        self.current_image = load_image(str(path))

        override = self.overrides.get(path.name)
        if override is not None:
            self.threshold_offset_var.set(override["threshold_offset"])
            self.min_object_size_var.set(override["min_object_size"])
        # else: sliders keep whatever value they already had, carried over
        # from the previous image -- usually a reasonable starting point
        # since lighting is similar across one batch.

        self.filename_label.configure(
            text=f"{path.name}  ({self.current_index + 1} / {len(self.image_paths)})"
        )
        self.status_label.configure(text="")
        self._update_nav_buttons()
        self._recompute_and_redraw()

    # ---- segmentation + display --------------------------------------------

    def _current_segment_kwargs(self) -> dict:
        return {
            "threshold_offset": self.threshold_offset_var.get(),
            "min_object_size": int(self.min_object_size_var.get()),
        }

    def _current_dish_diameter(self) -> float:
        try:
            return self.dish_diameter_var.get()
        except tk.TclError:
            return DEFAULT_DISH_DIAMETER_MM

    def _on_param_change(self, _arg=None) -> None:
        if self.current_index >= 0:
            self._recompute_and_redraw()

    def _recompute_and_redraw(self) -> None:
        image = self.current_image
        seg = segment_colony(image, **self._current_segment_kwargs())
        cal = calibrate(image, dish_diameter_mm=self._current_dish_diameter())
        metrics = compute_all_metrics(seg.colony_mask, seg.gray, mm_per_pixel=cal.mm_per_pixel)

        self.ax.clear()
        draw_overlay_on_axes(self.ax, image, seg.colony_mask, cal)
        self.ax.axis("off")
        self.canvas.draw_idle()

        self._update_metrics_panel(metrics, cal)

    def _update_metrics_panel(self, metrics: dict, cal: CalibrationResult) -> None:
        unit = "mm" if cal.calibrated else "px"
        lines = [
            f"Area: {metrics['area']:.1f} {unit}^2",
            f"Perimeter: {metrics['perimeter']:.1f} {unit}",
            f"Equiv. diameter: {metrics['equivalent_diameter']:.1f} {unit}",
            f"Circularity: {metrics['circularity']:.3f}",
            f"Solidity: {metrics['solidity']:.3f}",
            f"Texture contrast: {metrics['texture_contrast']:.2f}",
            f"Texture entropy: {metrics['texture_entropy']:.2f}",
            "",
            f"Calibrated: {'yes' if cal.calibrated else 'no (px units)'}",
        ]
        if cal.warning:
            lines.append(f"Warning: {cal.warning}")

        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.configure(state=tk.DISABLED)

    # ---- actions ------------------------------------------------------------

    def accept_parameters(self) -> None:
        path = self.image_paths[self.current_index]
        self.overrides[path.name] = self._current_segment_kwargs()
        self.status_label.configure(text=f"Saved parameters for {path.name}")

    def run_batch_export(self) -> None:
        output_dir = filedialog.askdirectory(
            title="Choose output folder for measurements.csv",
            initialdir=str(self.input_dir.parent) if self.input_dir else None,
        )
        if not output_dir:
            return

        self.status_label.configure(text="Running batch...")
        self.root.update_idletasks()

        try:
            df = run_batch(
                self.input_dir,
                output_dir,
                dish_diameter_mm=self._current_dish_diameter(),
                segment_kwargs=self._current_segment_kwargs(),
                overrides=self.overrides,
            )
        except Exception as exc:
            self.status_label.configure(text="Batch failed.")
            messagebox.showerror("Batch failed", str(exc))
            return

        message = f"Processed {len(df)} image(s).\nResults written to {output_dir}/measurements.csv"
        self.status_label.configure(text=message)
        messagebox.showinfo("Batch complete", message)


def main() -> None:
    """Launch the segmentation review GUI."""
    root = tk.Tk()
    root.geometry("900x650")
    ColonyReviewApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
