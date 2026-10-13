# RobotDancing Project Website

This directory is a buildless static website for GitHub Pages. All asset paths are relative, so the site works at `/RobotDancing/` without a build step.

## Publish

1. Commit and push `docs/` and the repository `.gitignore` changes to `main`. Include `docs/assets/hero.mp4` and `docs/assets/main-video.mp4`.
2. In `BAAI-Humanoid/RobotDancing`, open **Settings > Pages**.
3. Select **Deploy from a branch**, branch **main**, folder **/docs**, and save.
4. Wait for the Pages deployment to complete. The expected URL is https://baai-humanoid.github.io/RobotDancing/.

`.nojekyll` enables direct static serving. No npm dependencies or GitHub Actions configuration are required.

## Media

- `assets/hero.mp4`: short muted background clip.
- `assets/main-video.mp4`: web-optimized complete video, with audio and fast-start MP4 metadata.
- `source/main_video.mp4`: local high-quality original; intentionally ignored by Git. The website never links to this file.

Keep hosted files below GitHub's 100 MB per-file limit. Avoid Git LFS for Pages media.

## Local Preview

Open `index.html` in a browser, or run `python3 -m http.server 8000 --directory docs` from the repository root and visit http://localhost:8000/.
