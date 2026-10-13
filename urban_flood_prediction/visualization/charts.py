import matplotlib.pyplot as plt
import pandas as pd

def rainfall_chart(df, district):
    d = df[df["District"] == district].sort_values("Date").tail(30)
    plt.figure(figsize=(10, 4))
    plt.plot(d["Date"], d["Rainfall_mm"])
    plt.title(f"Rainfall - {district}")
    plt.xlabel("Date")
    plt.ylabel("Rainfall (mm)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return plt.gcf()
