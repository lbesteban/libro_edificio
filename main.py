import sys
import os
import warnings
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

from config import (
    DOCS_DIR,
    OUTPUT_DIR,
    ALLOWED_EXTENSIONS,
    IGNORED_EXTENSIONS,
    BATCH_SIZE,
    PROVEEDOR,
    MODEL_NAME,
    API_KEY,
    API_URL,
    PAGES_PER_BATCH,
    ENABLE_FAST_LOCAL_PATH,
    get_group_filename
)
from processor import (
    CheckpointManager,
    DocumentProcessor,
    QuotaExhaustedError,
    TokenEstimator,
    get_vision_client
)


def run_pipeline():
    """Ejecuta el pipeline de consolidación optimizado con Fast-Path, lotes y estimación de costes."""
    print("=" * 75)
    print("PROCESADOR DEL LIBRO DEL EDIFICIO PARA NOTEBOOKLM (OPTIMIZADO)")
    print("=" * 75)
    print(f"Directorio origen    : {DOCS_DIR}")
    print(f"Directorio destino   : {OUTPUT_DIR}")
    print(f"Proveedor activo     : {PROVEEDOR}")
    print(f"Modelo seleccionado  : {MODEL_NAME}")
    print(f"Vía Rápida Local     : {'ACTIVADA (Fast-Path $0.00)' if ENABLE_FAST_LOCAL_PATH else 'DESACTIVADA'}")
    print(f"Lotes Multipágina    : {PAGES_PER_BATCH} páginas por llamada API")
    if API_URL:
        print(f"Endpoint URL         : {API_URL}")

    if not DOCS_DIR.exists():
        print(f"[ERROR] El directorio origen '{DOCS_DIR}' no existe.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Inicializar componentes
    checkpoint_mgr = CheckpointManager()
    
    if not API_KEY or API_KEY == "tu_api_key_aqui":
        print(f"[WARN] No se detectó {PROVEEDOR}_API_KEY válida en .env.")

    vision_client = get_vision_client(
        provider=PROVEEDOR,
        api_key=API_KEY,
        model_name=MODEL_NAME,
        api_url=API_URL
    )
    doc_processor = DocumentProcessor(vision_client, checkpoint_mgr)

    # Recolectar todos los archivos a procesar
    all_files = []
    for root, _, files in os.walk(DOCS_DIR):
        for f in files:
            file_path = Path(root) / f
            ext = file_path.suffix.lower()
            if ext in ALLOWED_EXTENSIONS and ext not in IGNORED_EXTENSIONS:
                all_files.append(file_path)

    total_files = len(all_files)
    print(f"Archivos válidos encontrados: {total_files}")
    print("=" * 75)

    if total_files == 0:
        print("No hay archivos que procesar.")
        return

    processed_count = 0
    skipped_count = 0
    quota_reached = False
    auto_confirm_all = False

    with tqdm(all_files, desc="Procesando documentos", unit="doc") as pbar:
        for file_path in pbar:
            if BATCH_SIZE > 0 and processed_count >= BATCH_SIZE:
                tqdm.write(f"\n[INFO] Límite de lote alcanzado ({BATCH_SIZE} archivos procesados). Finalizando.")
                break

            rel_path_obj = file_path.relative_to(DOCS_DIR)
            rel_path_str = str(rel_path_obj).replace("\\", "/")
            
            pbar.set_postfix_str(f"Fichero: {rel_path_obj.name[:25]}")

            if checkpoint_mgr.is_file_processed(rel_path_str):
                skipped_count += 1
                continue

            group_filename = get_group_filename(rel_path_obj)
            target_md_path = OUTPUT_DIR / group_filename

            # 1. Estimación detallada de tokens, discriminando Local vs API y calculando coste + markup 5%
            est = TokenEstimator.estimate_file(file_path, model_name=MODEL_NAME, provider=PROVEEDOR)
            
            tqdm.write("\n" + "-" * 75)
            tqdm.write(f"[ESTIMACIÓN & COSTE] {rel_path_str} ({est.num_pages} pág(s))")
            tqdm.write(f" - Páginas Texto Local (Fast-Path $0.00) : {est.local_pages} pág(s)")
            tqdm.write(f" - Páginas Visión API ({PROVEEDOR})         : {est.api_pages} pág(s) en {est.api_batches} lote(s)")
            if est.api_pages > 0:
                tqdm.write(f" - Tokens API Entrada (proyectados)     : ~{est.input_tokens:,} tokens")
                tqdm.write(f" - Tokens API Salida (proyectados)      : ~{est.output_tokens:,} tokens")
                tqdm.write(f" - Tarifa Modelo ({MODEL_NAME})          : ${est.price_input_1m}/1M In | ${est.price_output_1m}/1M Out")
                if est.markup_usd > 0:
                    tqdm.write(f" - Recargo OpenRouter (5% Mark-up)       : +${est.markup_usd:.6f} USD")
            tqdm.write(f" - COSTE TOTAL ESTIMADO                 : ~${est.total_cost_usd:.6f} USD")
            tqdm.write("-" * 75)

            # 2. Interacción Y / N / A (Yes / No / Always)
            if not auto_confirm_all:
                try:
                    ans = input("¿Continuar procesando este archivo? [Y]es / [N]o (detener) / [A]lways (no volver a preguntar) [Y]: ").strip().upper()
                except (EOFError, KeyboardInterrupt):
                    ans = "N"

                if ans == "N" or ans == "NO":
                    tqdm.write("\n[INFO] Ejecución detenida por la respuesta del usuario. Avances guardados en checkpoint.")
                    break
                elif ans == "A" or ans == "ALWAYS":
                    auto_confirm_all = True
                    tqdm.write("[INFO] Modo 'ALWAYS' (A) activado. Se mostrarán las estimaciones de costes pero no se solicitará más confirmación.")

            try:
                completed = doc_processor.process_file_incremental(
                    file_path=file_path,
                    rel_path=rel_path_str,
                    target_md_path=target_md_path,
                    group_filename=group_filename
                )
                if completed:
                    processed_count += 1

            except QuotaExhaustedError as qe:
                tqdm.write(f"\n\n[AVISO DE CUOTA] {qe}")
                tqdm.write("[INFO] Todos los avances han sido guardados en 'processed_files.json'.")
                tqdm.write("[INFO] Puedes volver a ejecutar 'main.py' en cualquier momento para continuar exactamente donde se quedó.\n")
                quota_reached = True
                break

    print("\n" + "=" * 75)
    print("RESUMEN DE LA EJECUCIÓN")
    print(f"- Proveedor / Modelo              : {PROVEEDOR} ({MODEL_NAME})")
    print(f"- Archivos procesados en esta sesión: {processed_count}")
    print(f"- Archivos previos omitidos (ckpt) : {skipped_count}")
    print(f"- Estado de la cuotas / Ejecución  : {'Límite alcanzado (Pausado)' if quota_reached else 'OK / Finalizado'}")
    print(f"- Archivos consolidados generados  : {OUTPUT_DIR}")
    print("=" * 75)


if __name__ == "__main__":
    run_pipeline()
