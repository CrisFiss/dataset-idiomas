# spa_ger a, o, h, e, n
# ger_eng e, o, n t, a
# eng_spa t, h, a, c, d

# spa e, a, o, s, n
# ger e, n, i, r, s
# eng e, t, a, o, s

from collections import Counter
import re

with open('test_eng.txt') as txt:
    texto_prueba = txt.read()

def indentificador_idioma(texto):
    letters = list(re.sub("[^a-z]+", "", texto.lower()))
    # COUNTING & SORTING
    sorted_dict = {k: v for k, v in sorted(dict(Counter(letters)).items(), key=lambda x: x[1])}

    # PERCENTAGES
    percentage = {}
    s = sum(sorted_dict.values())
    for k, v in sorted_dict.items():
        pct = v * 100.0 / s
        percentage[k] = pct
    print(percentage)
    print(sum(percentage.values()))

    idioma = ""
    for k, v in percentage.items():
        if percentage["a"] > percentage["o"] and percentage["o"] > percentage["s"]:
            idioma = "Espanol"
        elif percentage["n"] > percentage["i"] and percentage["i"] > percentage["r"]:
            idioma = "Aleman"
        elif percentage["t"] > percentage["a"] and percentage["a"] > percentage["o"]:
            idioma = "Ingles"
        else:
            idioma = "distinto del Espanol, Aleman o Ingles"
    print(f"El idioma es {idioma}")

indentificador_idioma(texto_prueba)

