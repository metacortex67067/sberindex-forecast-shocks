"""
Бюллетени Росстата «Денежные доходы и расходы населения» (.doc, таблицы Word) ->
data/raw/rosstat/region_income_spending.csv: регион × год × (среднедушевой доход, потребительские расходы на душу), руб./мес.
Требуется LibreOffice (soffice) для конвертации .doc -> .docx и пакет python-docx.
"""
import glob, os, subprocess, sys, tempfile
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from sbershock.config import RAW
from sbershock.data.loaders import _match_region, load_dictionary
import docx

TABLES = {"Таблица 3": "income_pc", "Таблица 4": "spending_pc"}
d = load_dictionary(); regions = sorted(d.region_name.unique())
code = d.drop_duplicates("region_name").set_index("region_name").region_code
rows = []
for folder in sorted(glob.glob(str(RAW / "rosstat" / "den_dohod" / "*"))):
    tmp = tempfile.mkdtemp()
    for tname, var in TABLES.items():
        src = glob.glob(os.path.join(folder, "**", f"razd1-{tname}.doc"), recursive=True)[0]
        subprocess.run(["soffice", "--headless", "--convert-to", "docx", "--outdir", tmp, src],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=180)
        T = docx.Document(os.path.join(tmp, os.path.basename(src) + "x")).tables[0]
        years = [c.text.strip() for c in T.rows[1].cells][1:3]
        for r in T.rows[2:]:
            cells = [c.text.strip() for c in r.cells]
            reg = _match_region(cells[0], regions) if cells[0] else None
            if cells[0] == "Российская Федерация":
                reg = "RF"
            if reg is None:
                continue
            for y, v in zip(years, cells[1:3]):
                v = pd.to_numeric(v.replace(" ", "").replace("\xa0", "").replace(",", "."), errors="coerce")
                rows.append((0 if reg == "RF" else int(code[reg]), int(y), var, v))
out = pd.DataFrame(rows, columns=["region_code", "year", "var", "value"]).drop_duplicates(
    ["region_code", "year", "var"], keep="last")  # более свежий выпуск уточняет данные
out = out.pivot_table(index=["region_code", "year"], columns="var", values="value").reset_index()
out.to_csv(RAW / "rosstat" / "region_income_spending.csv", index=False)
print(out.groupby("year").size().to_dict(), out[out.region_code == 0].to_string(index=False))
