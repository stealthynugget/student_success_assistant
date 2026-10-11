# Data

Place the seven official OULAD CSV files in `data/raw/` (not committed, not in Docker images):

`assessments.csv, courses.csv, studentAssessment.csv, studentInfo.csv, studentRegistration.csv, studentVle.csv, vle.csv`

Source: Kuzilek, Hlosta & Zdrahal (2017), *Open University Learning Analytics dataset*, Scientific Data 4:170171 —
https://analyse.kmi.open.ac.uk/open_dataset (CC-BY 4.0).

`python -m ssa.cli prepare` writes `data/processed/student_features.csv.gz` and `weekly_engagement.npz`.
`python -m ssa.cli demo-data` writes a small synthetic fixture with a `.synthetic_fixture` marker; the quality gate then
switches to the *smoke* profile and marks results as non-official.
