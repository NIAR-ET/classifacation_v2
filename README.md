# AM Porosity Analyzer (Streamlit single-file)

Streamlit app for Metal AM bead porosity checks. Upload individual images or a ZIP, run deterministic OpenCV analysis (no ML), and export Excel + sorted annotated images (ACCEPT / REJECT).

## Quick start
1. Install Python 3.10+ and dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```
3. In your browser, upload weld bead images (JPG/PNG/TIFF/BMP or ZIP), adjust settings, review ACCEPT/REJECT output, and export Excel + sorted images from the **Downloads** tab.

## Notes
- Deterministic computer vision (OpenCV) only—no training required.
- Excel report includes the criteria used (`ACCEPT if Porosity ≤ threshold`).
- Sorted ZIP splits annotated images into `ACCEPT/` and `REJECT/` folders with a corner mark for quick review.
