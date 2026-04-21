# Proyecto-de-IA-1

## Configurar entorno virtual de Python

1. Verifica que Python este instalado:

```powershell
python --version
```

2. Crea el entorno virtual en la raiz del proyecto:

```powershell
python -m venv .venv
```

3. Activa el entorno virtual:

- PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

- CMD:

```bat
.\.venv\Scripts\activate.bat
```

## Instalar dependencias desde requirements.txt

Con el entorno virtual activo, ejecuta:

```powershell
pip install -r requirements.txt
```

## Verificacion rapida

Para confirmar que las dependencias se instalaron correctamente:

```powershell
pip list
```

