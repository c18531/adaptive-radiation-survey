"""Extract the Kamenna deposition tables (Tests 5/3P, 6/4P) from IAEA-TECDOC-2001, Appendix I, Tables 29-30.

The IAEA data are not redistributed in this repository. Download the PDF yourself and run:
    python extract_kamenna.py TE-2001web.pdf
Source: https://www-pub.iaea.org/MTCD/Publications/PDF/TE-2001web.pdf
Requires the `pdftotext` command (poppler). Writes revision_R1/data_kamenna/kamenna_5_3P.csv and kamenna_6_4P.csv
and checks the expected row counts (168 and 158).
"""
import csv, re, subprocess, sys
from pathlib import Path
out = Path(__file__).resolve().parents[1] / "data_kamenna"; out.mkdir(exist_ok=True)
txt = subprocess.run(["pdftotext", "-layout", sys.argv[1], "-"], capture_output=True, text=True, check=True).stdout
L = txt.split("\n")
i29 = [i for i, l in enumerate(L) if l.strip().startswith("TABLE 29.")]
i30 = [i for i, l in enumerate(L) if l.strip().startswith("TABLE 30.")]
end = next(i for i in range(i30[-1], len(L)) if "arrangement of detectors" in L[i])
num = re.compile(r"-?\d+(\.\d+)?")
def grab(a, b):
    return [tuple(map(float, l.split())) for l in L[a:b] if len(l.split()) == 3 and all(num.fullmatch(x) for x in l.split())]
for name, rows, n in [("5_3P", grab(i29[0], i30[0]), 168), ("6_4P", grab(i30[0], end), 158)]:
    assert len(rows) == n, (name, len(rows))
    with open(out / f"kamenna_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["x_m", "y_m", "deposition_Bq_m2"]); w.writerows(rows)
    print(name, len(rows), "rows written")
