import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ------------------ MODEL ------------------
def model(x, g0, g1):
    return x * (1 - x) * (g0 + g1 * x)

def read_float_list(text):
    try:
        return [float(i.strip()) for i in text.split(",")]
    except:
        return None

# ------------------ UI ------------------
st.title("Excess Gibbs Energy Fit")

R = 8.31415

T = st.number_input("Temperature (K)", value=298.15)

x_input = st.text_input("Molar fraction (x)", "0.1,0.2,0.3")
p1_input = st.text_input("Solvent pressure (p1)", "10,20,30")
p2_input = st.text_input("Solute pressure (p2)", "5,15,25")

p1o = st.number_input("Solvent saturated vapor pressure", value=100.0)
p2o = st.number_input("Solute saturated vapor pressure", value=100.0)

# ------------------ COMPUTE ------------------
if st.button("Run fit"):

    x = read_float_list(x_input)
    p1 = read_float_list(p1_input)
    p2 = read_float_list(p2_input)

    if None in (x, p1, p2):
        st.error("Invalid input format")
    elif not (len(x) == len(p1) == len(p2)):
        st.error("All inputs must have same length")
    else:
        df = pd.DataFrame({"x": x, "p1": p1, "p2": p2})

        df['a1'] = df['p1'] / p1o
        df['a2'] = df['p2'] / p2o

        df['g1'] = df['a1'] / (1 - df["x"])
        df['g2'] = df['a2'] / df['x']

        df['Gex'] = R*T*((1-df['x'])*np.log(df['g1']) + df['x']*np.log(df['g2']))

        # Fit
        params, _ = curve_fit(model, df['x'], df['Gex'])
        g0, g1 = params

        # R^2
        y_pred = model(df['x'], g0, g1)
        r2 = 1 - np.sum((df['Gex'] - y_pred)**2) / np.sum((df['Gex'] - np.mean(df['Gex']))**2)

        st.write(f"**g0 = {g0:.5f}**")
        st.write(f"**g1 = {g1:.5f}**")
        st.write(f"**R² = {r2:.5f}**")
        st.write("Warning! If your R² is less than 0.95 this model is UNFIT!")

        # Plot
        X = np.linspace(0, 1, 100)

        fig, ax = plt.subplots()
        ax.plot(df['x'], df['Gex'], 'o', label='data')
        ax.plot(X, model(X, g0, g1), '--', label='fit')

        # Sign formatting (cleaner)
        sign = "+" if g1 >= 0 else "-"
        ax.text(0.5, 0, f"$G^{{ex}} = (1-x)x({g0:.3f} {sign} {abs(g1):.3f}x)$", ha='center')

        ax.grid('--', alpha=0.5)
        ax.set_xlabel('Solute mole fraction, $\\chi$')
        ax.set_ylabel('Excess Gibbs energy, $G^{ex}$ (J/mol)')
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_title('Excess Gibbs Energy')

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)
