## Contributing to BG_Flood

Thank you for your interest in contributing to BG_Flood! Whether you are fixing a bug, optimizing parallel loops, expanding hydrodynamic capabilities,
or improving documentation, your contributions help make this flood model more efficient and accessible for the scientific and engineering community.

As a project aimed at high-performance environmental modelling, we value technical accuracy, code performance, and clear documentation.

---

## Code of Conduct

By participating in this project, you agree to maintain a professional,
collaborative, and respectful environment. Focus discussions on technical merits, performance optimization, and scientific validity.

---

## How to Contribute

### 1. Reporting Bugs & Feature Requests

Before opening a new issue, please search the existing issue tracker to see
if the topic has already been discussed. If you find a new bug or have a feature request, open an issue including:

* **A clear, descriptive title.**
* **Environment details:** Operating system, compiler version, and relevant hardware details (e.g., CPU/GPU specs).
* **A minimal working example (MWE):** For bugs, provide a simplified configuration, a small Digital Elevation Model (DEM) snippet, or the exact simulation parameters (rainfall, sea level boundary conditions, etc.) required to reproduce the issue.
* **Expected vs. actual behavior.**

### 2. Developing Code Changes

We welcome code contributions that improve model physics, accelerate performance, or enhance file I/O operations.

#### Environment & Language Standards

* **Core Engine:** High-performance routines utilize parallel programming frameworks. Ensure any modifications to the code **try** to preserve parallel efficiency of the kernels
* **Data Formats:** BG_Flood relies heavily on structured grid structures and geospatial formats. Ensure that any updates affecting coordinate tracking or spatial ordering correctly handle standard vertical datums and projections without loss of spatial fidelity.
* **Minimal Dependencies:** BG_Flood is meant to be as portable as possible and should only rely on minimal dependancies. Only significant features should call for adding dependancies  

#### Code Style Guide

* Keep functions modular and well-commented, particularly when implementing complex hydraulic or hydrodynamic equations.
* Optimize arrays and memory access patterns for continuous block memory layouts to maximize cache efficiency.
* Consider the readability trade-off when optimising
* 

---

## Pull Request Process

To submit your changes, please follow these steps:

1. **Fork the Repository:** Create a personal fork of the BG_Flood repository.
2. **Create a Branch:** Use a descriptive branch name (e.g., `feature/numba-raster-optimization` or `fix/boundary-condition-leak`).
3. **Implement Tests:** If adding a new feature or fixing a hydraulic routing bug, include corresponding **internal** validation tests. Verify your changes against standard benchmark flood scenarios if modifying any of the hydraulic engines.
4. **Document Changes and show your own test results:** Update any relevant markdown documentation, inline comments, or configuration schema definitions if your changes introduce new parameters (e.g., modified rainfall ARI inputs or custom roughness coefficients).
5. **Submit the PR:** Make sure you stage your changes appropriately. Well tested features That succeeds all of its test can open a Pull Request against `development` branch but poorly optimise experiment that need a bit of work should pull request on another branch off developments
6. **No PR to main:** PR to main are reserved for lead dev team and will only be merged after thourough checks from the development branch. Exceptions are for text edit of documentation. 

### Pull Request Checklist

* [ ] Code compiles both with  without warnings using the project's standard compiler flags.
* [ ] Results match expected outputs within acceptable floating-point tolerances.
* [ ] Spatial data structures properly handle boundary cells and edge cases.
* [ ] The documentation has been updated to reflect the changes.

---

## Community & Academic Attribution

BG_Flood is open-source scientific software. If your contributions are substantial or introduce a novel hydrodynamic modeling methodology, they may be eligible for co-authorship attribution in future software papers (such as submissions to the Journal of Open Source Software (JOSS)). Please ensure your commit history uses a consistent name and email address for proper attribution.
