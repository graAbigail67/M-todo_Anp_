"""
Módulo ANP (Analytic Network Process)
Funciones matemáticas para el método ANP
"""

import numpy as np
import pandas as pd

# ==========================================================
# ÍNDICES ALEATORIOS DE SAATY
# ==========================================================

RI = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49
}

# ==========================================================
# CREAR MATRIZ DE SAATY
# ==========================================================

def crear_matriz(n):
    """Crea una matriz identidad de tamaño n para Saaty."""
    return np.eye(n)


# ==========================================================
# LLENAR MATRIZ DE SAATY
# ==========================================================

def llenar_matriz(matriz, comparaciones):
    """Llena la matriz de Saaty con las comparaciones por pares."""
    n = len(matriz)
    indice = 0
    
    for i in range(n):
        for j in range(i + 1, n):
            valor = comparaciones[indice]
            matriz[i, j] = valor
            matriz[j, i] = 1.0 / valor
            indice += 1
    
    return matriz


# ==========================================================
# SUMA DE COLUMNAS
# ==========================================================

def suma_columnas(matriz):
    """Calcula la suma de cada columna de la matriz."""
    return matriz.sum(axis=0)


# ==========================================================
# MATRIZ NORMALIZADA
# ==========================================================

def normalizar_matriz(matriz):
    """Normaliza la matriz dividiendo cada elemento por la suma de su columna."""
    suma = suma_columnas(matriz)
    
    if np.any(suma == 0):
        raise ValueError("Error: Una columna suma 0, no se puede normalizar.")
    
    return matriz / suma


# ==========================================================
# VECTOR DE PRIORIDADES
# ==========================================================

def vector_prioridades(matriz_normalizada):
    """Calcula el vector de prioridades como promedio de cada fila."""
    return matriz_normalizada.mean(axis=1)


# ==========================================================
# LAMBDA MÁXIMO
# ==========================================================

def lambda_max(matriz, prioridades):
    """Calcula el valor propio máximo (λ max)."""
    if np.any(prioridades == 0):
        raise ValueError("Error: El vector de prioridades contiene ceros.")
    
    producto = np.dot(matriz, prioridades)
    cociente = producto / prioridades
    return np.mean(cociente)


# ==========================================================
# ÍNDICE DE CONSISTENCIA (CI)
# ==========================================================

def indice_consistencia(lambda_maximo, n):
    """Calcula el Índice de Consistencia (CI)."""
    if n <= 1:
        return 0.0
    return (lambda_maximo - n) / (n - 1)


# ==========================================================
# RAZÓN DE CONSISTENCIA (CR)
# ==========================================================

def razon_consistencia(ci, n):
    """Calcula la Razón de Consistencia (CR)."""
    if n <= 2:
        return 0.0
    return ci / RI.get(n, 1.49)


# ==========================================================
# VALIDAR CONSISTENCIA
# ==========================================================

def validar_consistencia(cr):
    """Verifica si la matriz es consistente (CR ≤ 0.10)."""
    return cr <= 0.10


# ==========================================================
# RESUMEN DE RESULTADOS AHP
# ==========================================================

def resumen_resultados(matriz):
    """Calcula todos los resultados AHP de una matriz de comparación."""
    n = len(matriz)
    
    normalizada = normalizar_matriz(matriz)
    prioridades = vector_prioridades(normalizada)
    lm = lambda_max(matriz, prioridades)
    ci = indice_consistencia(lm, n)
    cr = razon_consistencia(ci, n)
    consistente = validar_consistencia(cr)
    
    return {
        "matriz": matriz,
        "normalizada": normalizada,
        "prioridades": prioridades,
        "lambda": lm,
        "ci": ci,
        "cr": cr,
        "consistente": consistente
    }


# ==========================================================
# CONSTRUIR SUPERMATRIZ ANP
# ==========================================================

def construir_supermatriz(criterios, prioridades_principales, dependencias):
    """
    Construye la supermatriz ANP a partir de las dependencias entre criterios.
    Retorna ndarray (matriz estocástica por columnas).
    """
    n = len(criterios)
    supermatriz = np.zeros((n, n))
    indice = {c: i for i, c in enumerate(criterios)}
    
    # Llenar con dependencias
    for objetivo in dependencias:
        df = dependencias[objetivo]
        columna = indice[objetivo]
        
        for _, fila in df.iterrows():
            criterio = fila["Criterio"]
            peso = fila["Peso"]
            
            if criterio in indice:
                fila_indice = indice[criterio]
                supermatriz[fila_indice, columna] = peso
    
    # Normalizar cada columna para que sume 1
    for j in range(n):
        suma = supermatriz[:, j].sum()
        if suma > 0:
            supermatriz[:, j] = supermatriz[:, j] / suma
        else:
            supermatriz[j, j] = 1.0
    
    return supermatriz


# ==========================================================
# SUPERMATRIZ PONDERADA
# ==========================================================

def supermatriz_ponderada(supermatriz):
    """Calcula la supermatriz ponderada normalizando por columnas."""
    matriz = supermatriz.copy()
    
    for j in range(matriz.shape[1]):
        suma = matriz[:, j].sum()
        if suma > 0:
            matriz[:, j] = matriz[:, j] / suma
    
    return matriz


# ==========================================================
# SUPERMATRIZ LÍMITE
# ==========================================================

def supermatriz_limite(supermatriz, tolerancia=1e-8, max_iter=1000):
    """
    Calcula la supermatriz límite mediante potencias sucesivas.
    W^(k+1) = W @ W^k
    """
    W = supermatriz.copy()
    W_k = W.copy()
    
    for _ in range(max_iter):
        W_nueva = W @ W_k
        error = np.max(np.abs(W_nueva - W_k))
        W_k = W_nueva
        
        if error < tolerancia:
            break
    
    return W_k


# ==========================================================
# RANKING FINAL ANP
# ==========================================================

def ranking_final(supermatriz_limite, criterios=None):
    """Calcula el ranking final a partir de la supermatriz límite."""
    pesos = supermatriz_limite[:, 0]
    
    suma_pesos = pesos.sum()
    if abs(suma_pesos - 1.0) > 0.001:
        pesos = pesos / suma_pesos
    
    if criterios is None:
        criterios = [f"Criterio {i+1}" for i in range(len(pesos))]
    
    ranking = pd.DataFrame({
        "Criterio": criterios,
        "Peso": pesos
    })
    
    ranking = ranking.sort_values(by="Peso", ascending=False).reset_index(drop=True)
    ranking["Posición"] = range(1, len(ranking) + 1)
    ranking = ranking[["Posición", "Criterio", "Peso"]]
    
    return ranking