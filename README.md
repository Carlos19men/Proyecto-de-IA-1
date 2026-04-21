# Proyecto de Análisis y Predicción de Niveles de Ríos

Este proyecto se enfoca en el análisis de datos históricos y la creación de modelos predictivos para los niveles de los ríos, utilizando la metodología CRISP-DM para organizar el flujo de trabajo.

## Estructura del Proyecto (CRISP-DM)

El repositorio está organizado siguiendo las fases de la metodología CRISP-DM para asegurar un desarrollo estructurado y escalable.

-   `data/`: Almacena todos los conjuntos de datos utilizados en el proyecto.
    -   `data/raw/`: Datos brutos sin procesar.
    -   `data/processed/`: Datos limpios y transformados, listos para el modelado.
-   `notebooks/`: Contiene los Jupyter Notebooks para la exploración, análisis y experimentación.
-   `src/`: Código fuente modularizado.
    -   `src/data_understanding/`: Scripts para la carga y exploración inicial de datos.
    -   `src/data_preparation/`: Scripts para la limpieza, transformación y preparación de datos.
    -   `src/modeling/`: Scripts para la construcción y entrenamiento de modelos.
    -   `src/evaluation/`: Scripts para la evaluación de los modelos.
-   `models/`: Almacena los modelos entrenados y serializados.
-   `reports/`: Contiene los informes, visualizaciones y resultados generados.

## Configurar Entorno Virtual de Python

1.  **Verifica que Python esté instalado**:
    ```powershell
    python --version
    ```
2.  **Crea el entorno virtual**:
    ```powershell
    python -m venv .venv
    ```
3.  **Activa el entorno virtual**:
    -   PowerShell:
        ```powershell
        .\.venv\Scripts\Activate.ps1
        ```
    -   CMD:
        ```bat
        .\.venv\Scripts\activate.bat
        ```

## Instalar Dependencias

Con el entorno virtual activo, instala las librerías necesarias:

```powershell
pip install -r requirements.txt
```

## Verificación Rápida

Para confirmar que las dependencias se instalaron correctamente, puedes listar los paquetes:

```powershell
pip list
```

