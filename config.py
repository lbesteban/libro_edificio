import os
import re
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno desde .env
load_dotenv()

# Base path
BASE_DIR = Path(__file__).parent.resolve()

# Rutas principales
DOCS_DIR = Path(os.getenv("DOCS_DIR", r"C:\src\libro_edificio\docs")).resolve()
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", r"C:\src\libro_edificio\md_consolidado")).resolve()
CHECKPOINT_FILE = BASE_DIR / "processed_files.json"

# Selector de Proveedor y Modelo
PROVEEDOR = os.getenv("PROVEEDOR", os.getenv("MODELO", "GEMINI")).upper()

def _get_provider_var(prov: str, var_suffix: str, default_val: str = "") -> str:
    """Busca {PROVEEDOR}_{SUFFIX}, fallback a GEMINI_{SUFFIX} o {SUFFIX}."""
    primary_var = f"{prov}_{var_suffix}"
    val = os.getenv(primary_var)
    if val and val.strip():
        return val.strip()
    fallback_gemini = f"GEMINI_{var_suffix}"
    val_gemini = os.getenv(fallback_gemini)
    if val_gemini and val_gemini.strip():
        return val_gemini.strip()
    return os.getenv(var_suffix, default_val).strip()

MODEL_NAME = _get_provider_var(PROVEEDOR, "MODEL", "google/gemini-2.5-flash")
API_KEY = _get_provider_var(PROVEEDOR, "API_KEY", "")
API_URL = _get_provider_var(PROVEEDOR, "URL", "")

# Filtrado estricto
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
IGNORED_EXTENSIONS = {".dwg", ".bak", ".log", ".map", ".pc3", ".ctb"}

# Parámetros de rendimiento y lotes
RENDER_DPI = int(os.getenv("RENDER_DPI", "100"))  # 100 DPI es óptimo para OCR rápido y poco tamaño
MIN_TEXT_CHARS = 120
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "0"))  # 0 = sin límite por lote de archivos
PAGES_PER_BATCH = int(os.getenv("PAGES_PER_BATCH", "10"))  # Páginas por llamada API multipágina
ENABLE_FAST_LOCAL_PATH = os.getenv("ENABLE_FAST_LOCAL_PATH", "true").lower() in ("true", "1", "yes")
DELAY_BETWEEN_REQUESTS = float(os.getenv("DELAY_BETWEEN_REQUESTS", "1.0"))  # Segundos entre llamadas API

# Tarifas oficiales OpenRouter (USD por 1.000.000 tokens: (input, output))
OPENROUTER_PRICING = {
    "google/gemini-2.5-flash": (0.075, 0.30),
    "google/gemini-1.5-flash": (0.075, 0.30),
    "google/gemini-2.0-flash-exp": (0.10, 0.40),
    "google/gemini-1.5-pro": (1.25, 5.00),
    "deepseek/deepseek-chat": (0.14, 0.28),
    "deepseek/deepseek-v3": (0.14, 0.28),
    "deepseek/deepseek-r1": (0.55, 2.19),
    "openai/gpt-4o-mini": (0.15, 0.60),
    "openai/gpt-4o": (2.50, 10.00),
    "anthropic/claude-3.5-haiku": (0.80, 4.00),
    "anthropic/claude-3.5-sonnet": (3.00, 15.00),
    "moonshot/kimi-v1-8k": (0.20, 0.50),
}

# Prompt Multimodal del Sistema
SYSTEM_PROMPT = """Actúa como un perito arquitectónico y documentalista técnico de máxima precisión especializado en el "Libro del Edificio".

Analiza las páginas/imágenes adjuntas y genera una transcripción y descripción estructurada en Markdown técnico limpio.

Instrucciones estrictas:
1. **Documentos técnicos, fichas y boletines**:
   - Transcribe con exactitud absoluta tablas de calidades, marcas, modelos, números de serie, fechas y sellos oficiales.
   - Preserva el formato de tabla Markdown (`| Columna | ... |`).

2. **Planos y Esquemas de Instalaciones**:
   - Describe estancias, cotas clave, trazados de tuberías, conductos, cuadros eléctricos, materiales, diámetros y leyendas.

3. **Formato de Salida**:
   - Devuelve ÚNICAMENTE Markdown sintáctico impecable.
   - NO incluyas introducciones conversacionales ni preámbulos.
"""

def sanitize_name(name: str) -> str:
    """Convierte una cadena en un nombre de archivo seguro sin caracteres especiales."""
    name = re.sub(r'([A-Za-z])\s*\.\s*(\d+)', r'\1\2', name)
    name = re.sub(r'[^\w\d_]', '_', name)
    name = re.sub(r'_+', '_', name).strip('_')
    return name

def get_group_filename(relative_path: Path) -> str:
    """Determina el nombre del archivo Markdown consolidado (.md) según la subcarpeta."""
    parts = relative_path.parts
    if len(parts) == 1:
        return "General_Libro_Edificio.md"
    
    if parts[0].lower() == "anexos":
        if len(parts) >= 3 and any(parts[1].startswith(prefix) for prefix in ["B.", "C.", "D.", "E.", "F."]):
            group_folder = parts[2]
            return f"{sanitize_name(group_folder)}.md"
        elif len(parts) >= 2:
            group_folder = parts[1]
            return f"{sanitize_name(group_folder)}.md"
            
    return f"{sanitize_name(parts[0])}.md"
