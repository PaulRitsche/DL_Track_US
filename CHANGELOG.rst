
2025-11-23
==========

Fixed
-----

- Qeueing in UI to speed up image loading.

2025-11-18
==========

Added
-----

- Tangetial extrapolation of Aponeurosis in each direction
- Faster PA calculation
- PA to results plot in Main UI
- pyproject.toml / setup.cfg / setup.py: Bump version to 0.3.1

Fixed
-----

- MT calculation based on middle image region
- Fascicle intersection with muscle apo
- Manual Video analysis loading

2025-06-02
==========

Added
-----

- versioning for mkdocs page with mike

Changed
-------

- Documentation: for manual and automatic image and video analysis to include new models in tutorial.

Fixed
-----

- pyproject.toml / setup.cfg / setup.py: Bump version to 0.3.0
- mkdocs.yml

2025-05-26
==========

Removed
-------

- Old docs
- Figures Folder
- Test Folder (included in online docs)

Added
-----

- DL_Track mkdocs documentation

- Api docs
- Logo and Icons to main UI

- Logo in advanced_methods

Fixed
-----

- slow video loading and processing

- Documentation updates for the upcoming v0.3.0 release.
- Error handling in advancec_methods.py
- Resizing in main UI and advancec_methods.py

2025-03-11
==========

Added
-----

- Curved fascicle reconstruction to main UI

- Video Cropping in UI advanced window
- Video Part removal in UI advanced window
- "Live" preview of the segmented video in the main UI.
- Plotting operation of fascicles in the main UI.
- Manual Scaling in main UI for images and video.

Changed
-------

- UI to customtkinter finalisation
- Dictionary handling
- Settings.py file

- Migrated whole UI to customtkinter
- Manual Scaling in main UI for images and video.

2024-05-07
==========

Added
-----

- curved_fascicle.py file for curvature analysis.

2023-10-13
==========

Added
-----

- Apo length threshold for prediction

Changed
-------

- GUI layout in Analysis window to add apo length threshold

Fixed
-----

- Error Hhandling for apo prediction

2023-10-09
==========

Added
-----

- Mask inspection inside GUI
- Version update to 0.2.1
- Fascicle filtering to avoid overlapping fascicles

Changed
-------

- GUI layout and handling of analysis mode switches
