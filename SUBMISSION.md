# Zidio submission — Project FORESIGHT (Eswar Mahalingam)

| # | Item | Marks | Link / file |
|---|---|---|---|
| 1 | Source code | 5 | GitHub: `<paste repo URL>` — this folder pushed as-is |
| 2 | Live deployment | 5 | https://foresight-northbay.netlify.app · API: https://foresight-northbay.netlify.app/api/score?sku=SKU012 |
| 3 | Demo video | 4 | `<unlisted YouTube link>` — script: reports/Video_Scripts_Demo_and_Feedback.pdf (section A) |
| 4 | Feedback video | 4 | `<unlisted YouTube link>` — script: section B |
| 5 | Project report | 2 | reports/Project_Report_FORESIGHT.pdf (+ D7_Executive_Readout.pdf, D2_EDA_Data_Quality_Memo.pdf) |

## Push to GitHub (2 minutes)
```bash
cd foresight
git init && git add . && git commit -m "Project FORESIGHT — demand & inventory intelligence (Zidio DS internship)"
gh repo create foresight --public --source=. --push        # if GitHub CLI is installed
# otherwise: create an empty repo named foresight on github.com, then
git branch -M main && git remote add origin https://github.com/<you>/foresight.git && git push -u origin main
```
## Redeploy after changes
```bash
python src/run_all.py && netlify deploy --prod --dir=web   # site id e763cf3a-b370-48b8-a4a3-b08d7d6f4a90 (foresight-northbay)
```
