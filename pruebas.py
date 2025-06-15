import pandas as pd
import matplotlib.pyplot as plt

# Cargar o crear DataFrame
df = pd.read_csv("wine_dataset.csv")  # o df = pd.DataFrame(...)

# Obtener solo las primeras filas (cabecera)
header_df = df.head(8)
print(df.shape)

# Crear figura
fig, ax = plt.subplots(figsize=(len(header_df.columns)*1.2, 2))  # ancho depende de columnas
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=header_df.values,
                 colLabels=header_df.columns,
                 cellLoc='center',
                 loc='center')

# Guardar como imagen
plt.savefig("wines_experiment/wines_head.pdf", dpi=300, bbox_inches='tight')
plt.close()