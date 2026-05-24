algorithm = "TOFL"
FIGURE_SIZE=(14, 10)
FONTSIZE=24
LEGEND_FONTSIZE=18

legends_dicts = { "en": {"random": "Random",
                         "m_fastest": "M-Fastest (M=50%)",
                         "tofl_oracle": "Oracle",
                         "tofl_estimator_dl" : "TOFL-e",
                         "tofl": "TOFL-e",
                         "tofl_estimator_m_fastest": "TOFL-s (M=50%)",
                         "tofl_estimator_m_fastest_clients": "TOFL-s (M=50%)",
                         "tofl_mfastest": "TOFL-s (M=50%)"},

                "pt":   {"random": "Aleatório",
                         "m_fastest": "M-Fastest (M=50%)",
                         "tofl_oracle": algorithm+" Oráculo",
                         "tofl_estimator_dl" : algorithm+" Estimador",
                         "tofl": algorithm+" Estimador",
                         "tofl_estimator_m_fastest": algorithm+" com M-Fastest",
                         "tofl_estimator_m_fastest_clients": algorithm+" com M-Fastest",
                         "tofl_mfastest": algorithm+" com M-Fastest"} 
                         
                         }

style = {"random": "-",
         "m_fastest": "--",
         "tofl_oracle": "-.",
         "tofl": "-",
         "tofl_estimator_dl": "-",
         "tofl_estimator_m_fastest": (0, (1, 3)),
         "tofl_estimator_m_fastest_clients": (0, (1, 3)),
         "tofl_mfastest":(0, (3, 10, 1, 10))}

colors = {"random": "b",
          "m_fastest": "r",
          "tofl_oracle": "y",
          "tofl": "k",
          "tofl_estimator_dl": "k",
          "tofl_estimator_m_fastest": "g",
          "tofl_estimator_m_fastest_clients": "g",
          "tofl_mfastest":"gray"}
