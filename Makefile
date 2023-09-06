.PHONY: install run_model run_detection

install:
	pip install opencv-python pillow numpy

run_model: install
	python model.py

run_detection: run_model
	python camera_detection.py
