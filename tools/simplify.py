import glob
import os
import sys
import time

import numpy as np
import pydicom
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from radstract.data.dicom import (
    convert_dicom_to_images,
    convert_images_to_dicom,
)


class CropSelectorDialog(QDialog):
    # This dialog still outputs the standard bounding box (x1, y1, x2, y2)
    # The conversion to (x1, y1, width, height) will happen in DicomConverterGUI
    def __init__(self, dicom_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Crop Coordinates")
        self.setGeometry(100, 100, 800, 600)

        self.crop_coords = None

        # Load DICOM and get middle frame using convert_dicom_to_images
        try:
            # We must load the DICOM first to get dimensions for the viewer
            ds = pydicom.dcmread(dicom_path)

            images = convert_dicom_to_images(ds)

            # Get middle image
            middle_idx = len(images) // 2
            image = images[middle_idx]

            # Convert PIL Image to QPixmap
            # First convert to numpy array
            # Ensure the image is correctly converted (handling possible PIL specific formats)
            img_array = np.array(
                image.convert("L")
                if image.mode != "RGB" and image.mode != "RGBA"
                else image
            )

            # Handle RGB vs grayscale
            if len(img_array.shape) == 3:
                # RGB or RGBA image
                height, width, channels = img_array.shape
                bytes_per_line = channels * width
                if channels == 3:
                    # Using QImage.Format_RGB888 ensures correct interpretation of data buffer
                    q_image = QImage(
                        img_array.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGB888,
                    )
                elif channels == 4:
                    q_image = QImage(
                        img_array.data,
                        width,
                        height,
                        bytes_per_line,
                        QImage.Format_RGBA8888,
                    )
                else:
                    raise ValueError(f"Unsupported channel count: {channels}")
            else:
                # Grayscale image (L mode in PIL)
                height, width = img_array.shape
                bytes_per_line = width
                q_image = QImage(
                    img_array.data.tobytes(),  # Ensure data is byte buffer
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format_Grayscale8,
                )

            # Make a copy because QImage might rely on the numpy array data buffer
            self.pixmap = QPixmap.fromImage(q_image.copy())

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"Failed to load DICOM for preview: {str(e)}"
            )
            self.reject()
            return

        # Layout
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Click and drag to select crop region. "
            "Coordinates shown are (x1, y1, x2, y2)."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Graphics view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setDragMode(QGraphicsView.NoDrag)
        layout.addWidget(self.view)

        # Add image to scene
        self.scene.addPixmap(self.pixmap)

        # Rectangle item for selection
        self.rect_item = None
        self.start_point = None
        self.is_drawing = False

        # Coordinate display
        self.coord_label = QLabel("Coordinates (x1, y1, x2, y2): Not selected")
        layout.addWidget(self.coord_label)

        # Buttons
        button_layout = QHBoxLayout()
        self.reset_btn = QPushButton("Reset Selection")
        self.reset_btn.clicked.connect(self.reset_selection)
        button_layout.addWidget(self.reset_btn)

        self.ok_btn = QPushButton("OK")
        self.ok_btn.clicked.connect(self.accept)
        self.ok_btn.setEnabled(False)
        button_layout.addWidget(self.ok_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(self.cancel_btn)

        layout.addLayout(button_layout)

        # Install event filter on view
        self.view.viewport().installEventFilter(self)

    def eventFilter(self, obj, event):
        if obj == self.view.viewport():
            if event.type() == event.Type.MouseButtonPress:
                if event.button() == Qt.LeftButton:
                    self.start_drawing(event.position())
                    return True
            elif event.type() == event.Type.MouseMove:
                if self.is_drawing:
                    self.update_drawing(event.position())
                    return True
            elif event.type() == event.Type.MouseButtonRelease:
                if event.button() == Qt.LeftButton and self.is_drawing:
                    self.finish_drawing(event.position())
                    return True

        return super().eventFilter(obj, event)

    def start_drawing(self, pos):
        # Map to scene coordinates
        scene_pos = self.view.mapToScene(pos.toPoint())
        self.start_point = scene_pos
        self.is_drawing = True

        # Remove old rectangle if exists
        if self.rect_item:
            self.scene.removeItem(self.rect_item)

        # Create new rectangle
        pen = QPen(QColor(255, 0, 0), 2)
        # Initialize rectangle at the start point
        self.rect_item = QGraphicsRectItem(QRectF(scene_pos, scene_pos))
        self.rect_item.setPen(pen)
        self.scene.addItem(self.rect_item)

    def update_drawing(self, pos):
        if not self.is_drawing or not self.start_point:
            return

        scene_pos = self.view.mapToScene(pos.toPoint())

        # Update rectangle
        rect = QRectF(self.start_point, scene_pos).normalized()
        self.rect_item.setRect(rect)

    def finish_drawing(self, pos):
        self.is_drawing = False

        if not self.start_point:
            return

        scene_pos = self.view.mapToScene(pos.toPoint())

        # Get final rectangle
        rect = QRectF(self.start_point, scene_pos).normalized()

        # Clamp coordinates to image bounds
        image_rect = self.pixmap.rect()
        x1 = max(0, int(rect.left()))
        y1 = max(0, int(rect.top()))
        x2 = min(image_rect.width(), int(rect.right()))
        y2 = min(image_rect.height(), int(rect.bottom()))

        # Ensure valid crop (min size 1x1)
        if x2 <= x1 or y2 <= y1:
            self.reset_selection()
            QMessageBox.warning(
                self,
                "Invalid Selection",
                "Selection area must have positive width and height.",
            )
            return

        # Store coordinates (x1, y1, x2, y2)
        self.crop_coords = (x1, y1, x2, y2)

        # Update display
        self.coord_label.setText(
            f"Coordinates (x1, y1, x2, y2): {x1}, {y1}, {x2}, {y2}"
        )
        self.ok_btn.setEnabled(True)

        # Redraw the final clamped rectangle
        final_rect = QRectF(x1, y1, x2 - x1, y2 - y1)
        self.rect_item.setRect(final_rect)

    def reset_selection(self):
        if self.rect_item:
            self.scene.removeItem(self.rect_item)
            self.rect_item = None

        self.start_point = None
        self.is_drawing = False
        self.crop_coords = None
        self.coord_label.setText("Coordinates (x1, y1, x2, y2): Not selected")
        self.ok_btn.setEnabled(False)

    def get_coordinates(self):
        """Returns (x1, y1, x2, y2)"""
        return self.crop_coords


class WorkerThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    error_occurred = Signal(str)
    processing_complete = Signal()

    def __init__(
        self,
        input_path,
        output_path,
        glob_pattern,
        nested,
        crop_coords_internal,
        compress_factor,
        is_single_file,
    ):
        super().__init__()
        self.input_path = input_path
        self.output_path = output_path
        self.glob_pattern = glob_pattern
        self.nested = nested
        # crop_coords_internal is (x1, y1, width, height)
        self.crop_coords_internal = crop_coords_internal
        self.compress_factor = compress_factor
        self.is_single_file = is_single_file

    def run(self):
        try:
            if self.is_single_file:
                dicom_files = [self.input_path]
                self.log_updated.emit(
                    f"Processing single file: {self.input_path}"
                )
            else:
                os.makedirs(self.output_path, exist_ok=True)
                pattern = os.path.join(self.input_path, self.glob_pattern)

                if self.nested:
                    dicom_files = glob.glob(pattern, recursive=True)
                else:
                    dicom_files = glob.glob(pattern, recursive=False)

                if not dicom_files:
                    self.log_updated.emit(
                        f"No files found matching pattern: {pattern}"
                    )
                    self.status_updated.emit("No files found")
                    self.processing_complete.emit()
                    return

                self.log_updated.emit(
                    f"Found {len(dicom_files)} files to process"
                )

            # Process files
            file_count = len(dicom_files)
            for i, dicom_file in enumerate(dicom_files):
                self.status_updated.emit(
                    f"Processing: {os.path.basename(dicom_file)}"
                )
                # Update progress
                progress = int(((i + 1) / file_count) * 100)
                self.progress_updated.emit(progress)

                # Log every file processed, but maybe not on every progress update
                if i % 10 == 0 or i == file_count - 1:
                    self.log_updated.emit(
                        f"Processing ({i+1}/{file_count}): {dicom_file}"
                    )

                try:
                    ds = pydicom.dcmread(dicom_file)

                    images = convert_dicom_to_images(
                        ds,
                        crop_coordinates=self.crop_coords_internal,
                        compress_factor=self.compress_factor,
                    )
                    new = convert_images_to_dicom(images)

                    # Determine output file path
                    if self.is_single_file:
                        base_path, ext = os.path.splitext(dicom_file)
                        output_file = f"{base_path}-uncompressed{ext}"
                    else:
                        rel_path = os.path.relpath(dicom_file, self.input_path)
                        output_file = os.path.join(self.output_path, rel_path)

                        os.makedirs(
                            os.path.dirname(output_file), exist_ok=True
                        )

                    new.save_as(output_file)
                    self.log_updated.emit(f"Saved: {output_file}")
                except Exception as e:
                    self.log_updated.emit(
                        f"Error processing {dicom_file}: {str(e)}"
                    )

            # Final progress update
            self.progress_updated.emit(100)
            self.status_updated.emit(
                f"Completed processing {len(dicom_files)} files"
            )

        except Exception as e:
            self.error_occurred.emit(f"Worker Error: {str(e)}")

        self.processing_complete.emit()


class DicomConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Converter")
        self.setGeometry(
            100, 100, 600, 550
        )  # Increased height for new controls

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Input Selection ---
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input:"))
        # Defaulting to current dir (./) might not be ideal, but keeping previous behavior
        self.input_edit = QLineEdit("./")
        self.input_edit.textChanged.connect(self.on_input_changed)
        input_layout.addWidget(self.input_edit)
        input_browse_file_btn = QPushButton("Browse File")
        input_browse_file_btn.clicked.connect(self.browse_input_file)
        input_layout.addWidget(input_browse_file_btn)
        input_browse_folder_btn = QPushButton("Browse Folder")
        input_browse_folder_btn.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(input_browse_folder_btn)
        main_layout.addLayout(input_layout)

        # --- Output Folder Selection ---
        output_layout = QHBoxLayout()
        self.output_label = QLabel("Output Folder:")
        output_layout.addWidget(self.output_label)
        self.output_folder_edit = QLineEdit("./out")
        output_layout.addWidget(self.output_folder_edit)
        self.output_browse_btn = QPushButton("Browse")
        self.output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(self.output_browse_btn)
        main_layout.addLayout(output_layout)

        # --- Glob pattern & Nested ---
        pattern_layout = QHBoxLayout()
        self.pattern_label = QLabel("Glob Pattern:")
        pattern_layout.addWidget(self.pattern_label)
        self.glob_pattern_edit = QLineEdit("*.dcm")
        pattern_layout.addWidget(self.glob_pattern_edit)
        self.nested_checkbox = QCheckBox("Search in nested folders")
        self.nested_checkbox.setChecked(True)
        pattern_layout.addWidget(self.nested_checkbox)
        main_layout.addLayout(pattern_layout)

        # --- Compression Factor ---
        compress_layout = QHBoxLayout()
        compress_layout.addWidget(QLabel("Compression Factor (1.0 = None):"))
        self.compress_factor_spinbox = QDoubleSpinBox()
        self.compress_factor_spinbox.setRange(1, 10.0)
        self.compress_factor_spinbox.setSingleStep(1)
        self.compress_factor_spinbox.setValue(1.0)
        self.compress_factor_spinbox.setDecimals(0)
        compress_layout.addWidget(self.compress_factor_spinbox)
        main_layout.addLayout(compress_layout)

        # --- Crop coordinates ---
        crop_layout = QVBoxLayout()
        crop_layout.addWidget(
            QLabel(
                "Crop Coords (x1, y1, x2, y2 for reference; internally converted to x1, y1, W, H):"
            )
        )
        crop_input_layout = QHBoxLayout()
        self.crop_coords_edit = QLineEdit("")

        # Make the instruction label for crop coordinates slightly smaller or less intrusive
        crop_input_layout.addWidget(self.crop_coords_edit)
        self.select_crop_btn = QPushButton("Select Visually")
        self.select_crop_btn.clicked.connect(self.open_crop_selector)
        crop_input_layout.addWidget(self.select_crop_btn)
        crop_layout.addLayout(crop_input_layout)
        main_layout.addLayout(crop_layout)

        # --- Process Button ---
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_button)

        # --- Progress Bar & Status ---
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # --- Log Text Area ---
        main_layout.addWidget(QLabel("Processing Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        # Worker thread handle
        self.worker = None

        # Initialize UI state
        self.on_input_changed()

    def browse_input_file(self):
        file, _ = QFileDialog.getOpenFileName(
            self,
            "Select DICOM File",
            self.input_edit.text(),
            "DICOM Files (*.dcm *.DCM);;All Files (*)",
        )
        if file:
            self.input_edit.setText(file)

    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self.input_edit.text()
        )
        if folder:
            self.input_edit.setText(folder)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_folder_edit.text()
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def on_input_changed(self):
        """Enable/disable output folder selection based on input type"""
        input_path = self.input_edit.text().strip()
        is_file = os.path.isfile(input_path)

        # Output controls disabled if single file
        self.output_label.setEnabled(not is_file)
        self.output_folder_edit.setEnabled(not is_file)
        self.output_browse_btn.setEnabled(not is_file)

        # Pattern and nested controls disabled if single file
        self.pattern_label.setEnabled(not is_file)
        self.glob_pattern_edit.setEnabled(not is_file)
        self.nested_checkbox.setEnabled(not is_file)

        # Enable/Disable visual crop selection button based on if path looks valid
        self.select_crop_btn.setEnabled(os.path.exists(input_path))

    def open_crop_selector(self):
        """Open the visual crop coordinate selector"""
        input_path = self.input_edit.text()

        dicom_file = None

        if os.path.isfile(input_path):
            dicom_file = input_path
        elif os.path.isdir(input_path):
            pattern = os.path.join(input_path, self.glob_pattern_edit.text())
            nested = self.nested_checkbox.isChecked()
            # Use glob.glob to get the list of matching files
            files = glob.glob(pattern, recursive=nested)

            if files:
                # Use the first file found for visualization
                dicom_file = files[0]
            else:
                QMessageBox.warning(
                    self,
                    "No DICOM Files",
                    "No DICOM files found in the selected directory matching the pattern.",
                )
                return
        else:
            QMessageBox.warning(
                self,
                "Invalid Input",
                "Please select a valid DICOM file or folder first.",
            )
            return

        # Check if the file actually exists (important if input_path was manually typed)
        if not os.path.exists(dicom_file):
            QMessageBox.critical(
                self, "Error", f"DICOM file not found: {dicom_file}"
            )
            return

        # Open the crop selector dialog
        dialog = CropSelectorDialog(dicom_file, self)
        if dialog.exec() == QDialog.Accepted:
            # coords is (x1, y1, x2, y2)
            coords = dialog.get_coordinates()
            if coords:
                # Update the crop coordinates field
                coord_str = (
                    f"{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}"
                )
                self.crop_coords_edit.setText(coord_str)
                self.log_message(
                    f"Crop coordinates (x1, y1, x2, y2) selected: {coord_str}"
                )

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"{timestamp} - {message}")

    def start_processing(self):
        input_path = self.input_edit.text()
        is_single_file = os.path.isfile(input_path)

        if self.worker and self.worker.isRunning():
            QMessageBox.warning(
                self, "Busy", "Processing is already running. Please wait."
            )
            return

        # Validate inputs
        if not os.path.exists(input_path):
            QMessageBox.critical(self, "Error", "Input path does not exist")
            return

        output_path = self.output_folder_edit.text()
        if not is_single_file and not output_path:
            QMessageBox.critical(
                self,
                "Error",
                "Output folder path cannot be empty when processing a directory.",
            )
            return

        # --- 1. Parse and Convert Crop Coordinates ---
        crop_coords_text = self.crop_coords_edit.text().strip()
        crop_coords_internal = None  # format (x1, y1, width, height)

        if crop_coords_text:
            try:
                # Read selection (x1, y1, x2, y2)
                coords_bbox = tuple(map(int, crop_coords_text.split(",")))
                if len(coords_bbox) != 4:
                    raise ValueError(
                        "Crop coordinates must be 4 comma-separated integer values (x1, y1, x2, y2)."
                    )

                x1, y1, x2, y2 = coords_bbox

                # Validation
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        "x2 must be greater than x1 and y2 must be greater than y1."
                    )

                # Convert to required internal format: (x1, y1, width, height)
                width = x2 - x1
                height = y2 - y1
                crop_coords_internal = (x1, y1, width, height)

                self.log_message(
                    f"Converted crop coordinates for processing: {crop_coords_internal} (x1, y1, W, H)"
                )

            except ValueError as e:
                QMessageBox.critical(
                    self, "Error", f"Invalid crop coordinates format: {e}"
                )
                return

        # --- 2. Get Compression Factor ---
        compress_factor = self.compress_factor_spinbox.value()
        self.log_message(f"Using compression factor: {compress_factor}")

        # Disable button during processing
        self.process_button.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("Starting process...")
        self.log_text.clear()

        # Create and start worker thread
        self.worker = WorkerThread(
            input_path,
            output_path,
            self.glob_pattern_edit.text(),
            self.nested_checkbox.isChecked(),
            crop_coords_internal,
            compress_factor,
            is_single_file,
        )

        # Connect signals
        self.worker.progress_updated.connect(self.progress_bar.setValue)
        self.worker.status_updated.connect(self.status_label.setText)
        self.worker.log_updated.connect(self.log_message)
        self.worker.error_occurred.connect(self.show_error)
        self.worker.processing_complete.connect(self.processing_finished)

        # Start the thread
        self.worker.start()

    def show_error(self, error_message):
        QMessageBox.critical(self, "Execution Error", error_message)
        self.processing_finished()  # Ensure UI resets even on error

    def processing_finished(self):
        # Re-enable button
        self.process_button.setEnabled(True)
        self.status_label.setText("Processing finished.")

        # Clean up worker reference
        if self.worker:
            self.worker.wait()  # Wait for the thread to fully exit
            self.worker = None


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DicomConverterGUI()
    window.show()
    sys.exit(app.exec())
