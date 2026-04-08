# SLAM: Simultaneous Localization and Mapping

## Introduction

This project implements a complete SLAM pipeline to simultaneously estimate robot pose and build an environment map. The system fuses data from three sensors: wheel encoders, an inertial measurement unit (IMU), and a 2D LIDAR. A particle filter estimates the robot's trajectory through odometry integration, while a log-odds occupancy grid incrementally builds a probabilistic map of the environment by interpreting LIDAR measurements.

The core SLAM algorithm in `./scripts/run_slam.py` works in four steps:

1. **Odometry** – Integrate encoder and IMU data to predict particle positions
2. **Occupancy Grid** – Superimpose LIDAR scans onto particles to build log-odds maps
3. **Weight Update** – Score particles based on measurement likelihood
4. **Resampling** – Keep high-confidence particles, resample stragglers

## Code Structure

The project is organized into clear, modular sections:

```
 data/              # RAW SENSOR DATA. Contains encoder ticks, IMU readings, and LIDAR scans
 	train/         # Training set (three rooms: 20, 21, 23)
 	test/          # Test set
 
 src/slam_project/  # CORE IMPLEMENTATION. Well-organized as Python modules
 	io/            # Data loading utilities
 	localization/  # Odometry and particle filter
 	mapping/       # Mapper, raycasting, and scan matching
 	models/        # Motion model, measurement model, occupancy grid
 	visualization/ # Plotting and animation tools
 
 utils/             # HELPER UTILITIES
 	MapUtils/      # Pure Python raycasting
 	MapUtilsCython/ # Optimized Cython raycasting for speed
 
 scripts/           # HIGH-LEVEL WORKFLOWS. Run these to execute components!!!
 	run_slam.py    # Full SLAM pipeline (odometry + occupancy grid + particle filter)
 	run_odometry.py # Odometry only (visualize trajectory)
 	run_occupancy_grid.py # Occupancy grid only (single-particle map)
 	run_mapping.py # Mapping with known poses (oracle scenario)
 	evaluate_dataset.py # Batch evaluation across multiple datasets
 
 notebooks/         # INTERACTIVE DEBUGGING. Jupyter notebooks for debugging
 	debug_slam.ipynb
 	debug_occupancy_grid.ipynb
 	debug_mapping.ipynb
 
 outputs/           # RESULTS AND ARTIFACTS
 	figures/       # Generated plots and maps
 
 tests/             # More debugging, just not notebooks...
 
 pyproject.toml     # Dependency management (UV-based)
 README.md          # You are here!
```

## Setup Guide

Like the previous project, this uses UV for dependency management. After cloning the repository, set up and run:

```bash
uv run ./scripts/run_slam.py
```

This loads sensor data from the training set and executes the full SLAM pipeline for a single room, then generates visualizations in `outputs/`.