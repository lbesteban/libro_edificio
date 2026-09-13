# Procesador del Libro del Edificio para NotebookLM

Herramienta modular CLI e interactiva (Jupyter Notebook) diseñada para extraer, procesar y consolidar documentos técnicos del **Libro del Edificio** (PDFs digitales, escaneados e imágenes) generando archivos Markdown estructurados por subcarpetas optimizados para **NotebookLM**.

## Características Clave

- **Vía Rápida Local (Fast-Path PyMuPDF)**: Procesa páginas con texto vectorial nativo a coste **$0.00 USD** en milisegundos directamente en la CPU.
- **Lotes Multipágina**: Empaqueta hasta 10 páginas escaneadas en una sola llamada API, reduciendo las peticiones en más de un 90%.
- **Soporte Multiproveedor**: Compatible nativamente con **OpenRouter** y **Google Gemini**, extensible a OpenAI, DeepSeek, Kimi, Z.AI, etc.
- **Estimador de Tokens & Tarifas**: Muestra antes de cada documento la estimación de tokens de entrada/salida y coste en USD (incluyendo el 5% de recargo de OpenRouter) con flujo interactivo `[Y]es / [N]o / [A]lways`.
- **Checkpointing Granular**: Persistencia incremental por página y archivo en `processed_files.json` para reanudar ejecuciones interrumpidas sin duplicar el consumo de saldo.

## Requisitos

- Python 3.10+
- Clave de API de OpenRouter o Google Gemini

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu-usuario/libro_edificio.git
   cd libro_edificio
   ```

2. Crear e instanciar el entorno virtual:
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Linux/macOS
   # En Windows: venv\Scripts\activate
   ```

3. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```

4. Configurar el archivo `.env`:
   Copia la plantilla `.env.example` como `.env` e introduce tus claves:
   ```env
   DOCS_DIR=C:\src\libro_edificio\docs
   OUTPUT_DIR=C:\src\libro_edificio\md_consolidado

   PROVEEDOR=OPENROUTER
   OPENROUTER_MODEL=google/gemini-2.5-flash
   OPENROUTER_API_KEY=tu_api_key_aqui
   ```

## Uso

### Vía CLI (Línea de comandos)
```bash
python main.py
```

### Vía Jupyter Notebook
Abre [notebook.ipynb](notebook.ipynb) en tu editor o entorno Jupyter y ejecuta las celdas secuencialmente.

## Estructura del Proyecto

```text
libro_edificio/
├── config.py           # Configuración, rutas, tarifas y prompts
├── processor.py        # Clientes multimodales, Fast-Path y estimador de tokens
├── main.py             # CLI ejecutable con barra de progreso tqdm e interacción Y/N/A
├── notebook.ipynb      # Notebook equivalente interactivo
├── requirements.txt    # Dependencias del proyecto
├── .env.example        # Plantilla de configuración de entorno
└── .gitignore          # Archivos excluidos de control de versiones
```
