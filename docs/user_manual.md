# User manual

pypirisk tells you how likely a Python package is to stop being maintained in the next 18 months, and why. A high score means the package may need help. It is not a judgement of its maintainers.

Open http://localhost:8501 in a browser. The menu on the left has four pages.

## Grades

| Grade | Chance of being abandoned |
| --- | --- |
| A | under 10% |
| B | 10% to 20% |
| C | 20% to 35% |
| D | 35% to 50% |
| F | 50% or more |

## Pages

- Package Lookup: check one package.
  - Type a package name, for example `requests`, and press Enter.
  - You see its risk (the chance it is abandoned in the next 18 months) and its grade.
  - "Why" lists the three facts that affected the score most, for example "no release for 26 months raises the risk".
  - "Context" shows when it was last released, how active its code is, how many people do most of the work (bus factor), and how many packages depend on it.
  - The "Score live" switch recomputes the score now on a model server, instead of showing the last batch score.
- Project Scan: check a whole project.
  - Upload your `requirements.txt` (up to 100 KB) or paste package names, one per line, and press "Scan project".
  - The project grade comes from your riskiest dependency. Dependencies further down the tree count less, because a problem there is less likely to reach you.
  - "Expected dependencies abandoned" is how many of your dependencies are likely to be abandoned in total.
  - "Riskiest paths" shows how you depend on a risky package, for example `aiohttp -> async-timeout`.
  - The dependency graph colours each package by its grade.
- Ecosystem Watch: find packages that need support.
  - Risky packages that many others depend on are listed first.
  - Use the slider to show more rows, and "Minimum dependents" to hide packages that few others use.
- Pipeline Console: for the operations team.
  - The pipeline with each step green (last run worked) or red (failed).
  - How fast data was collected and checked.
  - Recent runs; turn on "Only failed runs" and open one to see the error.
  - The models in use and their history, with links to MLflow, Grafana, Prometheus and Alertmanager.

## Messages

| Message | What to do |
| --- | --- |
| "... is not in the scored package set" | The package has no score yet. Try Project Scan. |
| "That is not a valid PyPI package name." | Check the spelling. |
| "The risk service is unavailable" | Wait a minute. If it continues, tell the operations team. |
| "No batch scores yet; run the pipeline" | The operations team needs to run the pipeline once. |
