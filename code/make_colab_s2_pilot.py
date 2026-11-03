"""Build the Colab notebook of the LAT-S2 pilot with the four modules embedded; prints the code hash
(rule of LAT run.py: SHA-256 over sorted module names and their bytes, first 16 hex digits)."""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MODULES = ["norms.py", "model_io.py", "measures_s2.py", "run_pilot.py"]


def code_hash(srcs):
    h = hashlib.sha256()
    for name in sorted(srcs):
        h.update(name.encode())
        h.update(srcs[name].encode())
    return h.hexdigest()[:16]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": text}


def build(out_path):
    srcs = {m: open(os.path.join(HERE, m)).read() for m in MODULES}
    ch = code_hash(srcs)
    cells = [
        md("# LAT-S2, пилот: видит ли J-линза нормы в настоящей модели\n\n"
           "Разведка перед протоколом LAT-S2, выводы описательные. 12 норм × 2 предмета (свет, дверь); каждая норма "
           "при включении переворачивает ответ, при выключении оставляет его как в журнале. Монитор – J-линза "
           "(код Anthropic) на четверти, половине и трёх четвертях глубины модели. Меры как в LAT-S1b: Use, Seen, "
           "rel_R, классы «зрячий» / «слепой», ρc по формуле MIT, зонды применённого правила.\n\n"
           "**Правило решения:** идём в LAT-S2, если на одном из трёх слоёв или на первом читаемом слое (самом раннем, "
           "где линза уже читает факт) среди норм, которым модель следует, "
           "есть хотя бы 3 зрячих и 3 слепых. В основном расчёте, если модель следует меньше чем шести нормам, "
           "пилот останавливается до подгонки линзы; отладка проходит весь код в любом случае.\n\n"
           "**Как запускать.** 1) Отладка: среда T4, `MODE = \"debug\"` (Qwen2.5-1.5B, около 40 минут). "
           "2) Основной расчёт: среда A100, G4 или L4, `MODE = \"main\"` (Qwen2.5-7B; на A100 около 1,5 часа, "
           "на L4 около 3–4). Запустить все ячейки по порядку. После обрыва сессии – снова все ячейки: линза, "
           "температуры и готовые нормы подхватываются с Диска.\n\n"
           f"Хэш кода: `{ch}`."),
        code('MODE = "debug"          # "debug": T4, Qwen2.5-1.5B;  "main": A100 / G4 / L4, Qwen2.5-7B\n'
             'DRIVE_DIR = "/content/drive/MyDrive/LAT-S2/pilot"'),
        code('!pip install -q "jlens @ git+https://github.com/anthropics/jacobian-lens" datasets\n'
             'import torch, transformers, jlens\n'
             'print("torch", torch.__version__, "| transformers", transformers.__version__)\n'
             'print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "нет GPU",\n'
             '      f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB" if torch.cuda.is_available() else "")'),
        code('from google.colab import drive\n'
             'drive.mount("/content/drive")'),
        code("import os, hashlib\n"
             f"CODE = {json.dumps(srcs, ensure_ascii=False)}\n"
             'os.makedirs("/content/lat_s2_pilot", exist_ok=True)\n'
             "for name, src in CODE.items():\n"
             '    with open(os.path.join("/content/lat_s2_pilot", name), "w") as f:\n'
             "        f.write(src)\n"
             "h = hashlib.sha256()\n"
             "for name in sorted(CODE):\n"
             "    h.update(name.encode()); h.update(CODE[name].encode())\n"
             "CODE_HASH = h.hexdigest()[:16]\n"
             f'assert CODE_HASH == "{ch}", CODE_HASH\n'
             'print("code hash", CODE_HASH)'),
        code("import sys, json\n"
             'sys.path.insert(0, "/content/lat_s2_pilot")\n'
             "import run_pilot\n"
             "OUT = os.path.join(DRIVE_DIR, MODE)\n"
             "os.makedirs(OUT, exist_ok=True)\n"
             'open(os.path.join(OUT, "code_hash.txt"), "w").write(CODE_HASH)\n'
             "summary = run_pilot.main(MODE, OUT)"),
        code("s = summary\n"
             'print("ИТОГ:", "GO – идём в LAT-S2" if s["go"] else "STOP", s.get("stopped", ""))\n'
             'if s.get("stopped") == "preflight":\n'
             '    for r in s["preflight"]:\n'
             '        print(f\'{r["domain"]:6s} {r["norm"]:18s} точность без нормы {r["acc_R0"]:.2f}, с нормой {r["acc_R1"]:.2f}\')\n'
             "else:\n"
             '    for l, v in s["verdict"].items():\n'
             '        name = f"слой {l}" if l.isdigit() else "первый читаемый слой"\n'
             '        print(f"{name}: зрячих {v[\'n_sighted\']}, слепых {v[\'n_blind\']}, разведочно {v[\'exploratory\']}")\n'
             "    print()\n"
             '    ls = s["monitored"]\n'
             '    print(f\'{"предмет/норма":28s} {"следует":8s} \' + "  ".join(f"слой {l}: класс, rel, Seen_R, rho_c" for l in ls)\n'
             '          + "  первый читаемый: слой, класс, rel, Seen_R, rho_c")\n'
             '    for r in s["rows"]:\n'
             '        cells = []\n'
             "        for l in ls:\n"
             '            e = r[f"L{l}"]\n'
             '            f = lambda x: "  –  " if x is None or x != x else f"{x:5.2f}"\n'
             '            cells.append(f\'{e["cls"]:8s} {f(e["rel_main"])} {f(e["seen_R"])} {f(e["rho_c"])}\')\n'
             '        e = r["first_readable"]\n'
             '        fr = "  –" if e["layer"] is None else f\'{e["layer"]:3d} {e["cls"]:8s} {f(e["rel_main"])} {f(e["seen_R"])} {f(e["rho_c"])}\'\n'
             '        print(f\'{r["domain"] + "/" + r["norm"]:28s} {str(r["follows"]):8s} \' + "  ".join(cells) + "  " + fr)\n'
             'print("\\nФайлы на Диске:", OUT)'),
        md("**Что прислать мне после прогона:** файлы `summary.json` и `log.txt` из папки на Диске "
           "(`LAT-S2/pilot/debug` или `LAT-S2/pilot/main`). Остальное там же: линзы, температуры, "
           "результаты по каждой норме."),
    ]
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    with open(out_path, "w") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return ch


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "LAT_S2_pilot_colab.ipynb")
    print("code hash", build(out), "->", out)
