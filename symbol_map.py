SYMBOL_MAP = {
    "eq": "=",
    "minus": "-",
    "plus": "+",
    "times": "\\times",
    "divide": "/",
    "partialdiff": "\\partial",
    "product": "\\prod",
    "conditional": "|",
    "in": "\\in",
    "notin": "\\notin",
    "leq": "\\leq",
    "not-less-than-or-equals": "\\not \\leq",
    "geq": "\\geq",
    "neq": "\\neq",
    "less-than-and-not-equals": "\\lneq",
    "not-less-than-nor-greater-than": "\\nleq",
    "injective-limit": "\\varinjlim",
    "gt": ">",
    "lt": "<",
    "sum": "\\sum",
    "int": "\\int",
    "union": "\\bigcup",
    "intersect": "\\bigcap",
    "tensor-product": "\\otimes",
    "limit": "\\lim",
    "approaches-limit": "\\doteq",
    "similar-to": "\\sim",
    "less-than-or-similar-to": "\\lesssim",
    "limit-from": "",
    "supremum": "\\sup",
    "min": "\\min",
    "direct-sum": "\\oplus",
    "kernel": "\\ker",
    "exp": "\\exp",
    "exists": "\\exists",
    "pmod": "\\pmod",
    "degree": "\\deg",
    "max": "\\max",
    "ln": "\\ln",
    "or": "\\vee",
    "projective-limit": "\\varprojlim",
    "forall": "\\forall",
    "dimension": "\\dim",
    "arg": "\\arg",
    "imaginary": "\\Im",
    "coth": "\\coth",
    "log": "\\log",
    "sin": "\\sin",
    "cos": "\\cos",
    "tan": "\\tan",
    "cot": "\\cot",
    "sec": "\\sec",
    "arcsin": "\\arcsin",
    "arccos": "\\arccos",
    "arctan": "\\arctan",
    "sinh": "\\sinh",
    "cosh": "\\cosh",
    "tanh": "\\tanh",
    "approx": "\\approx",
    "not": "\\neg",
    "real": "\\Re",
    "det": "\\det",
    "maps-to": "\\mapsto",
    "coproduct": "\\coprod",
    "plus-or-minus": "\\pm",
    "minus-or-plus": "\\mp",
    "gcd": "\\gcd",
    "iff": "\\iff",
    "limit-infimum": "\\liminf",
    "compose": "\\circ",
    "quantum-operator-product": "|",
    "square-union": "\\sqcup",
    "assign": ":=",
    "contour-integral": "\\oint",
    "&quest;": "?",
    "limit-supremum": "\\limsup",
    "infimum": "\\inf",
    "much-less-than": "\\ll",
    "much-greater-than": "\\gg",
    "superset-of-or-equals": "\\supseteq",
    "superset-of": "\\supset",
    "subset": "\\subset",
    "prsubset": "\\subsetneq",
    "not-subset-of": "\\not \\subset",
    "not-subset-of-nor-equals": "\\nsubseteq",
    "not-subset-of-or-equals": "\\not \\subseteq",
    "square-image-of-or-equals": "\\sqsubseteq",
    "double-subset-of": "\\Subset",
    "direct-product": "\\odot",
    "left-factor-semidirect-product": "\\ltimes",
    "right-factor-semidirect-product": "\\rtimes",
    "similar-to-or-equals": "\\simeq",
    "not-similar-to-or-equals": "\\not \\simeq",
    "greater-than-or-equivalent-to": "\\gtrsim",
    "subgroup-of": "\\lhd",
    "contains-as-subgroup-or-equals": "\\unrhd",
    "subgroup-of-or-equals": "\\unlhd",
    "not-subgroup-of-nor-equals": "\\ntrianglelefteq",
    "perpendicular-to": "\\perp",
    "not-perpendicular-to": "\\not \\perp",
    "square-original-of-or-equals": "\\sqsupseteq",
    "proves": "\\vdash",
    "not-proves": "\\nvdash",
    "does-not-prove": "\\dashv",
    "models": "\\models",
    "not-models": "\\not \\models",
    "forces": "\\Vdash",
    "precedes": "\\prec",
    "precedes-or-equivalent-to": "\\precsim",
    "precedes-or-equals": "\\preccurlyeq",
    "not-divides": "\\nmid",
    "asymptotically-equals": "\\asymp",
    "proportional-to": "\\propto",
    "succeeds": "\\succ",
    "succeeds-or-equals": "\\succeq",
    "equivalent": "\\equiv",
    "not-equivalent-to": "\\not \\equiv",
    "bottom": "\\bot",
    "contains": "\\ni",
    "symmetric-difference": "\\ominus",
    "annotated": " ",
    "implies": "\\implies",
    "currency-dollar": "$",

    "infinity": "\\infty",
    "emptyset": "\\emptyset",
    "hbar": "\\hbar"
}

# Extends symbol map to include rough mappings for analyzing token embeddings
SYMBOL_MAP_ANALYSIS = {
    **SYMBOL_MAP,
    "differential-d": "\\,d",
    "percent": "%",
    "factorial": "!",
    "double-factorial": "!!",
    "root": "\\sqrt",
    "binomial": "\\binom",
    "continued-fraction": "\\cfrac",
    "SUB": "_",
    "SUP": "^"
}

# Mapping of some LaTeX symbols to unicode for display in visualizations
SYMBOL_MAP_DISPLAY = {
    "\\times": "×",
    "\\partial": "∂",
    "\\sum": "∑",
    "\\sqrt": "√",
    "\\in": "∈",
    "\\int": "∫",
    "\\leq": "≤",
    "\\sin": "sin",
    "\\cos": "cos",
    "\\log": "log",
    "\\geq": "≥",
    "\\ln": "ln",
    "\\approx": "≈",
    "\\otimes": "⊗",
    "\\binom": "()",
    "\\equiv": "≡",
    "\\exp": "exp",
    "\\circ": "◦",
    "\\cfrac": "/",
    "\\pm": "±",
    "\\subset": "⊂",
    "\\lim": "lim",
    "\\forall": "∀",
    "\\neq": "≠",
    "\\prod": "∏",
    "\\bigcup": "⋃",
    "\\sim": "∼",
    "\\oplus": "⊕",
    "\\neg": "¬",
    "\\tan": "tan",
    "\\bigcap": "⋂",
    "\\det": "det",
    "\\mapsto": "→",
    "\\max": "max",
    "\\vee": "∨",
    "\\exists": "∃",
    "\\min": "min",
    "\\pmod": "(mod )",
    "\\sup": "sup",
    "\\vdash": "⊢",
    "\\cosh": "cosh",
    "\\propto": "∝",
    "\\sinh": "sinh",
    "\\perp": "⊥",
    "\\cot": "cot",
    "\\arctan": "arctan",
    "\\oint": "∮"
}
