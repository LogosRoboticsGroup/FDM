# FDM project website

Static English project page for **Forward Dynamics Model**, with locally hosted fonts, figures, captions and H.264 video. No Node build or runtime dependencies are required.

The website lives in `docs/website/`, separate from the development guides in `docs/`; only the page and its prepared assets are deployed. Original PPTs and recordings remain local under `web/`.

## Preview

From the repository root:

```bash
python -m http.server 8000 --directory docs/website
```

Open `http://localhost:8000`. Use an HTTP server rather than opening `index.html` directly so captions and duration metadata load correctly.

## GitHub Pages

The project URL is https://logosroboticsgroup.github.io/FDM/.

1. Include `docs/website/index.html`, `docs/website/styles.css`, `docs/website/app.js`, **all of `docs/website/assets/`**, and `.github/workflows/project-page.yml` in the repository.
2. In repository **Settings → Pages → Build and deployment**, select **GitHub Actions**.
3. Push to the default branch, or run **Deploy FDM project page** manually from Actions. Only the default branch deploys automatically.
4. Use the successful deployment's `page_url` as the live URL. Creating these files locally does not publish the site.

The workflow uploads only the website and prepared assets. Raw recordings, PowerPoint files, media tooling, and local render intermediates are not deployed. All site assets use relative paths, including on `/FDM/`. If the repository moves, update the resource links, Open Graph image URL, and both main README links.

## Sources and editorial decisions

- Method, numerical results and figures: supplied `ICLR27_Forwar_Dynamics_Models.pdf` and figure PDFs in the matching ZIP.
- Public title: **Forward Dynamics Model** (singular), as specified by the author. The supplied PDF remains unmodified and uses the older plural title.
- Authors and institutions: repository README and manuscript.
- `gift/` contains the box-packing-and-closing experiment.
- Success rates are manuscript results, not newly measured results from this release. A clip's duration is not inference latency.
- `demo.mp4` is exported from the complete, updated 18-slide PowerPoint. Slides are rendered with LibreOffice; the five embedded comparison videos play in full at 1× speed, positioned in their original slide rectangles. PowerPoint build animations are flattened into static slide compositions. English captions are authored to explain each slide. A derived copy uses the requested singular title and omits the submission number; the original PPT is unchanged. No artificial voiceover or music is added.

## Media and captions

- `assets/demo.mp4`: clean overview with optional native English captions (`demo.en.vtt`, on by default).
- `assets/demo-captioned.mp4`: downloadable overview with English subtitles burned in.
- `assets/demo.en.srt`: editable subtitles for video editors.
- `assets/comparisons/`: 20 H.264/yuv420p synchronized composites with MP4 fast-start and WebP posters. Each is 640 × 720: third view at 640 × 480 above left/right wrist views at 320 × 240 each. Shorter views hold their final frame. Original videos remain locally in `web/real-exp/`.
- `assets/comparisons.json`: durations and any incomplete view names for each composite.
- `assets/*.{pdf,webp}`: original figure PDFs and raster derivatives.
- Fonts: DM Sans for body text, Libre Caslon Display for webpage headings, and Instrument Serif in the prepared video graphics, self-hosted under the SIL Open Font License; license files are in `assets/fonts/`.

To regenerate derived figures, videos and subtitles from source material (not needed for hosting):

```bash
pip install Pillow pypdfium2
# Install LibreOffice Impress and ffmpeg with libx264/libass support, then:
python docs/website/tools/prepare_media.py
```

The script uses the already downloaded fonts. Rendering intermediates stay in `playground/website/render/`. Keep `assets/comparisons.json` in sync. Poster and social-image generation is included in the media preparation script.

To re-export only the demo from an updated PPT, run `python docs/website/tools/export_ppt.py`. To rebuild only the 20 three-view comparisons, run `python docs/website/tools/compose_views.py`.
