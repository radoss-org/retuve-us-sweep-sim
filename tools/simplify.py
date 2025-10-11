import glob
import os
import sys
import time

import pydicom
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
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


class WorkerThread(QThread):
    progress_updated = Signal(int)
    status_updated = Signal(str)
    log_updated = Signal(str)
    error_occurred = Signal(str)
    processing_complete = Signal()

    def __init__(
        self, folder_path, output_path, glob_pattern, nested, crop_coords
    ):
        super().__init__()
        self.folder_path = folder_path
        self.output_path = output_path
        self.glob_pattern = glob_pattern
        self.nested = nested
        self.crop_coords = crop_coords

    def run(self):
        try:
            # Create output directory if it doesn't exist
            os.makedirs(self.output_path, exist_ok=True)

            # Base pattern for current folder
            pattern = os.path.join(self.folder_path, self.glob_pattern)

            if self.nested:
                # Recursive search through nested folders
                dicom_files = glob.glob(pattern, recursive=True)
            else:
                # Only current folder
                dicom_files = glob.glob(pattern, recursive=False)

            if not dicom_files:
                self.log_updated.emit(
                    f"No files found matching pattern: {pattern}"
                )
                self.status_updated.emit("No files found")
                self.processing_complete.emit()
                return

            self.log_updated.emit(f"Found {len(dicom_files)} files to process")

            # Process files
            for i, dicom_file in enumerate(dicom_files):
                self.status_updated.emit(f"Processing: {dicom_file}")
                self.log_updated.emit(f"Processing: {dicom_file}")

                # Update progress
                progress = int((i / len(dicom_files)) * 100)
                self.progress_updated.emit(progress)

                try:
                    # Use the DICOM file with your conversion functions
                    ds = pydicom.dcmread(dicom_file)

                    # Pass crop_coords if provided, otherwise None
                    crop_param = self.crop_coords if self.crop_coords else None
                    images = convert_dicom_to_images(
                        ds, crop_coordinates=crop_param
                    )
                    new = convert_images_to_dicom(images)

                    # Create output path maintaining relative structure
                    rel_path = os.path.relpath(dicom_file, self.folder_path)
                    output_file = os.path.join(self.output_path, rel_path)

                    # Create directory if it doesn't exist
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)

                    new.save_as(output_file)
                    self.log_updated.emit(f"Saved: {output_file}")
                except Exception as e:
                    self.log_updated.emit(
                        f"Error processing {dicom_file}: {str(e)}"
                    )

            # Complete progress
            self.progress_updated.emit(100)
            self.status_updated.emit(
                f"Completed processing {len(dicom_files)} files"
            )
            self.log_updated.emit(
                f"Completed processing {len(dicom_files)} files"
            )

        except Exception as e:
            self.error_occurred.emit(f"Error: {str(e)}")

        self.processing_complete.emit()


class DicomConverterGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM Converter")
        self.setGeometry(100, 100, 600, 500)

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # Input folder selection
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Input Folder:"))
        self.input_folder_edit = QLineEdit("./")
        input_layout.addWidget(self.input_folder_edit)
        input_browse_btn = QPushButton("Browse")
        input_browse_btn.clicked.connect(self.browse_input_folder)
        input_layout.addWidget(input_browse_btn)
        main_layout.addLayout(input_layout)

        # Output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Output Folder:"))
        self.output_folder_edit = QLineEdit("./out")
        output_layout.addWidget(self.output_folder_edit)
        output_browse_btn = QPushButton("Browse")
        output_browse_btn.clicked.connect(self.browse_output_folder)
        output_layout.addWidget(output_browse_btn)
        main_layout.addLayout(output_layout)

        # Glob pattern
        pattern_layout = QHBoxLayout()
        pattern_layout.addWidget(QLabel("Glob Pattern:"))
        self.glob_pattern_edit = QLineEdit("*.dcm")
        pattern_layout.addWidget(self.glob_pattern_edit)
        main_layout.addLayout(pattern_layout)

        # Nested search checkbox
        self.nested_checkbox = QCheckBox("Search in nested folders")
        self.nested_checkbox.setChecked(True)
        main_layout.addWidget(self.nested_checkbox)

        # Crop coordinates
        crop_layout = QVBoxLayout()
        crop_layout.addWidget(
            QLabel("Crop Coordinates (x1, y1, x2, y2) [optional]:")
        )
        self.crop_coords_edit = QLineEdit("")  # Empty default
        crop_layout.addWidget(self.crop_coords_edit)
        main_layout.addLayout(crop_layout)

        # Process button
        self.process_button = QPushButton("Process Files")
        self.process_button.clicked.connect(self.start_processing)
        main_layout.addWidget(self.process_button)

        # Progress bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # Status label
        self.status_label = QLabel("Ready")
        main_layout.addWidget(self.status_label)

        # Log text area
        main_layout.addWidget(QLabel("Processing Log:"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        main_layout.addWidget(self.log_text)

        # Worker thread
        self.worker = None

    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Input Folder", self.input_folder_edit.text()
        )
        if folder:
            self.input_folder_edit.setText(folder)

    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Folder", self.output_folder_edit.text()
        )
        if folder:
            self.output_folder_edit.setText(folder)

    def log_message(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"{timestamp} - {message}")

    def start_processing(self):
        # Validate inputs
        if not os.path.exists(self.input_folder_edit.text()):
            QMessageBox.critical(self, "Error", "Input folder does not exist")
            return

        # Parse crop coordinates - allow empty
        crop_coords_text = self.crop_coords_edit.text().strip()
        crop_coords = None

        if crop_coords_text:  # Only parse if not empty
            try:
                crop_coords = tuple(map(int, crop_coords_text.split(",")))
                if len(crop_coords) != 4:
                    raise ValueError("Crop coordinates must be 4 values")
            except ValueError as e:
                QMessageBox.critical(
                    self, "Error", f"Invalid crop coordinates: {e}"
                )
                return

        # Disable button during processing
        self.process_button.setEnabled(False)

        # Create and start worker thread
        self.worker = WorkerThread(
            self.input_folder_edit.text(),
            self.output_folder_edit.text(),
            self.glob_pattern_edit.text(),
            self.nested_checkbox.isChecked(),
            crop_coords,
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
        QMessageBox.critical(self, "Error", error_message)

    def processing_finished(self):
        # Re-enable button
        self.process_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DicomConverterGUI()
    window.show()
    sys.exit(app.exec())
