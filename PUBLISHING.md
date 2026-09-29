# Publishing checklist

The repository is prepared as a standalone local Git repository on the `main` branch. Review the rendered README and teaching notebook before creating the GitHub remote.

## 1. Confirm identity and release metadata

Check the author name in `LICENSE`, `CITATION.cff`, and `pyproject.toml`. Add an ORCID to `CITATION.cff` if desired. After GitHub creates the repository, add its final URL as `repository-code` in `CITATION.cff` and under `[project.urls]` in `pyproject.toml`.

Configure the Git identity that should appear on the initial commit:

```powershell
git config user.name "Your Name"
git config user.email "your-public-or-noreply-address"
git commit -m "Initial public release"
```

## 2. Create a private GitHub repository first

The GitHub CLI gives the clearest, repeatable flow:

```powershell
gh auth login
gh repo create flowsheet-rl-educational --private --source . --remote origin --push
```

Recommended description:

> Educational graph-based hierarchical PPO for synthetic process-flowsheet construction, with a step-by-step notebook and reproducible long-run example.

Recommended topics: `reinforcement-learning`, `process-systems-engineering`, `flowsheet-synthesis`, `graph-neural-networks`, `ppo`, `pytorch-geometric`, `education`.

VS Code’s **Publish Branch** command is also suitable after the local commit, but the CLI makes the repository name, visibility, remote, and push explicit.

## 3. Inspect the private remote

- Open the README and confirm all three figures render.
- Open `notebooks/learn_flowsheet_rl_step_by_step.ipynb` and confirm GitHub renders the cells and saved outputs.
- Confirm the PDF opens and the literature links resolve.
- Wait for the GitHub Actions test workflow to pass.
- Check that no `.venv`, `.runtime`, `runs`, cache, or `node_modules` content appears.
- Check the author identity and commit email shown by GitHub.

## 4. Make it public

Use GitHub **Settings → General → Danger Zone → Change repository visibility**, or run:

```powershell
gh repo edit --visibility public --accept-visibility-change-consequences
```

Then add an annotated `v0.1.0` tag and GitHub Release. The repository already contains the 2.3 MB introductory PDF. If you also want to distribute the 17 MB editable PowerPoint, attach it to the GitHub Release rather than adding it to normal source history.

## 5. After publication

- Add the final repository URL to the citation and package metadata.
- Create the `v0.1.0` release and archive its DOI with Zenodo if a citable research-software record is useful.
- Protect `main` and require the test workflow for pull requests if outside contributions are expected.
- Keep third-party publisher PDFs outside the repository; link DOI or authorized repository pages instead.
