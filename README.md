# GNSS Urban Environment Modeling and LOS/NLOS Area Dynamic Generation System

## Project Overview
This repository provides a complete, standalone simulation framework designed to model Global Navigation Satellite System (GNSS) signal propagation in complex urban environments. By combining parameterized 3D city modeling, real-world RINEX ephemeris data parsing, and geometric ray-tracing algorithms, this system automatically generates dynamic Line-of-Sight (LOS) and Non-Line-of-Sight (NLOS) signal classification matrices.

## Prerequisites and Installation
To run the modules in this repository, you need Python 3.8 or higher. The physical calculations and 3D geometric operations rely on several scientific computing libraries.

Install the required dependencies using pip:
```bash
pip install numpy trimesh matplotlib pandas xarray georinex rtree
```
Note: `rtree` is highly recommended as a spatial index backend for trimesh to significantly speed up ray-intersection queries.
You will also need standard `RINEX` navigation files (e.g., `hkcl0270.26n`). Place your ephemeris files in the root directory before running the advanced scripts.

## Module Descriptions and Execution Guide
This project is structured into five independent, progressive modules. Each script serves a distinct function and can be executed separately to observe different stages of the simulation pipeline.

### 1. `gnss_1.py`
This code simulates GNSS (GPS) signal visibility in an urban environment. It creates a simplified city block with buildings, then models satellite positions and uses ray tracing to determine at which ground points each satellite is visible (LOS) or blocked (NLOS) by buildings. The result is visualized as a 2D map showing LOS/NLOS regions for each satellite.

To run this module:
```bash
python gnss_1.py
```
### 2. `gnss_2.py`
This is an upgraded GNSS signal visibility simulator that now includes realistic GPS orbit calculations. Compared to the previous version, key improvements are:

Accurate satellite positioning: Implements standard GPS ephemeris formulas to compute actual ECEF coordinates from broadcast parameters, replacing the simplified azimuth/elevation model

Real coordinate conversion: Transforms satellite positions from global ECEF to local ENU coordinates for proper 3D ray tracing

Professional visualization: Generates comprehensive output with 5 detailed plots (3 projections, 3D view, visibility heatmap) plus raw data files

Enhanced simulation core: Fixed ray-tracing engine compatibility and added building projection in all views

Production-ready workflow: Separate data loading, calculation, and result export modules for scalability

The system now processes real GPS orbit data to determine exact satellite positions, calculates visibility at each ground point using ray casting, and produces complete visual documentation of NLOS/LOS patterns.

To run this module:
```bash
python gnss_2.py
```
### 3. `gnss_3.py`
This version now adds real-world RINEX file support through the georinex library, allowing direct processing of actual GPS broadcast ephemeris files instead of using hardcoded satellite parameters. It automatically reads all available satellites from navigation files, filters for GPS satellites, calculates their precise ECEF positions, and processes them in batch with automatic horizon filtering—making the simulation directly applicable to real GNSS data analysis.

To run this module:
```bash
python gnss_3.py
```
### 4. `gnss_4.py`
This version introduces multi-time-step dynamic simulations, a major upgrade from single snapshot analysis. Now the system generates sequential time-series outputs organized in structured folders (Step_00_Tplus_0min/, Step_01_Tplus_15min/, etc.) for each satellite over configurable time periods. This allows observing how GPS satellite visibility evolves over time (e.g., every 15 minutes) as satellites move along their orbits, creating a comprehensive temporal profile of urban canyon effects rather than just a static moment.

To run this module:
```bash
python gnss_4.py
```
### 5. `gnss_5.py`
This version now directly processes raw ephemeris records, analyzing all available time points in the RINEX file (typically covering a full day) rather than extrapolating from a single epoch. Instead of generating synthetic time steps, it outputs results for every actual broadcast moment (e.g., 6:00, 6:15, 6:30... throughout the day), organizing them in time-stamped folders. This provides real observation-based visibility data exactly matching satellite ground truth positions as broadcast, enabling validation against actual GPS measurements.

To run this module:
```bash
python gnss_5.py
```
## Conclusion
This repository offers a comprehensive framework for simulating GNSS signal propagation in urban environments, with progressive enhancements that incorporate real-world satellite data and dynamic temporal analysis. By following the execution guide for each module, users can explore the effects of urban canyons on GPS signal visibility and gain insights into LOS/NLOS classification in complex cityscapes. The final version provides a robust tool for researchers and practitioners working on GNSS signal analysis, urban planning, and related fields.
