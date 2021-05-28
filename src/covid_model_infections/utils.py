from pathlib import Path
import matplotlib.pyplot as plt

TIMELINE = {
    'deaths': 24,
    'cases': 11,
    'hospitalizations': 11,
}

# deaths/hosp limit cmooes from /home/j/WORK/12_bundle/lri_corona/9263/01_input_data/01_lit/mrbrt/final_asymptomatic_proportion_draws_by_age.csv
CEILINGS = {
    'deaths': 0.65,
    'cases': 0.80,
    'hospitalizations': 0.65,
}
