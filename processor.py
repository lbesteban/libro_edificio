import json
import time
import io
import re
import math
import base64
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from abc import ABC, abstractmethod
import pymupdf as fitz
from PIL import Image
import requests
from google import genai
from google.genai import types, errors

from config import (
    CHECKPOINT_FILE,
    ALLOWED_EXTENSIONS,
    IGNORED_EXTENSIONS,
    RENDER_DPI,
    MIN_TEXT_CHARS,
    PROVEEDOR,
    MODEL_NAME,
    API_KEY,
    API_URL,
    SYSTEM_PROMPT,
    PAGES_PER_BATCH,
    ENABLE_FAST_LOCAL_PATH,
    DELAY_BETWEEN_REQUESTS,
    OPENROUTER_PRICING
)


class QuotaExhaustedError(Exception):
    """Excepción para identificar cuando la cuota o saldo de la API se ha agotado totalmente."""
    pass


class TokenEstimation:
    def __init__(
        self,
        num_pages: int,
        local_pages: int,
        api_pages: int,
        api_batches: int,
        input_tokens: int,
        output_tokens: int,
        price_input_1m: float,
        price_output_1m: float,
        base_cost_usd: float,
        markup_usd: float,
        total_cost_usd: float
    ):
        self.num_pages = num_pages
        self.local_pages = local_pages
        self.api_pages = api_pages
        self.api_batches = api_batches
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.price_input_1m = price_input_1m
        self.price_output_1m = price_output_1m
        self.base_cost_usd = base_cost_usd
        self.markup_usd = markup_usd
        self.total_cost_usd = total_cost_usd


class TokenEstimator:
    """Calcula estimaciones de uso de tokens y costes discriminando Fast-Path local vs API con markup del 5%."""

    @classmethod
    def estimate_file(cls, file_path: Path, model_name: str = MODEL_NAME, provider: str = PROVEEDOR) -> TokenEstimation:
        ext = file_path.suffix.lower()
        num_pages = 1
        local_pages = 0
        api_pages = 0

        if ext == ".pdf":
            try:
                doc = fitz.open(file_path)
                num_pages = len(doc)
                for page in doc:
                    raw_text = page.get_text("text").strip()
                    images = page.get_images()
                    if ENABLE_FAST_LOCAL_PATH and len(raw_text) >= MIN_TEXT_CHARS:
                        local_pages += 1
                    else:
                        api_pages += 1
                doc.close()
            except Exception:
                num_pages = 1
                api_pages = 1
        elif ext in {".png", ".jpg", ".jpeg"}:
            num_pages = 1
            api_pages = 1

        api_batches = math.ceil(api_pages / PAGES_PER_BATCH) if api_pages > 0 else 0

        # Estimación de tokens solo para las páginas que van a la API
        input_tokens = (api_pages * 750) + (api_batches * 350) if api_pages > 0 else 0
        output_tokens = (api_pages * 400) if api_pages > 0 else 0

        # Obtener tarifa oficial del modelo
        pricing = OPENROUTER_PRICING.get(model_name.lower())
        if not pricing:
            # Buscar concordancia parcial
            for key, val in OPENROUTER_PRICING.items():
                if key in model_name.lower() or model_name.lower() in key:
                    pricing = val
                    break
        if not pricing:
            pricing = (0.15, 0.60)  # Tarifas promedio por defecto

        p_in, p_out = pricing
        base_cost = ((input_tokens / 1_000_000) * p_in) + ((output_tokens / 1_000_000) * p_out)

        # Aplicar Mark-up del 5% para OpenRouter
        markup = base_cost * 0.05 if provider.upper() == "OPENROUTER" else 0.0
        total_cost = base_cost + markup

        return TokenEstimation(
            num_pages=num_pages,
            local_pages=local_pages,
            api_pages=api_pages,
            api_batches=api_batches,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            price_input_1m=p_in,
            price_output_1m=p_out,
            base_cost_usd=base_cost,
            markup_usd=markup,
            total_cost_usd=total_cost
        )


class CheckpointManager:
    """Gestiona processed_files.json para reanudar a nivel de ARCHIVO Y PÁGINA INDIVIDUAL."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_FILE):
        self.checkpoint_path = checkpoint_path
        self.data: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        if self.checkpoint_path.exists():
            try:
                with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"[WARN] Error al cargar checkpoint ({e}), reiniciando de cero.")
                return {}
        return {}

    def is_file_processed(self, rel_path: str) -> bool:
        entry = self.data.get(rel_path)
        return entry is not None and entry.get("status") == "success"

    def is_page_processed(self, rel_path: str, page_num: int) -> bool:
        entry = self.data.get(rel_path)
        if not entry:
            return False
        if entry.get("status") == "success":
            return True
        completed_pages = entry.get("pages_completed", [])
        return page_num in completed_pages

    def mark_page_completed(self, rel_path: str, page_num: int, target_file: str, total_pages: int, save_disk: bool = True):
        if rel_path not in self.data:
            self.data[rel_path] = {
                "status": "in_progress",
                "target_file": target_file,
                "total_pages": total_pages,
                "pages_completed": []
            }

        pages = self.data[rel_path].setdefault("pages_completed", [])
        if page_num not in pages:
            pages.append(page_num)

        if len(pages) >= total_pages:
            self.data[rel_path]["status"] = "success"

        self.data[rel_path]["timestamp"] = time.time()
        if save_disk:
            self.save()

    def mark_file_processed(self, rel_path: str, target_file: str, total_pages: int = 1):
        self.data[rel_path] = {
            "status": "success",
            "target_file": target_file,
            "total_pages": total_pages,
            "pages_completed": list(range(1, total_pages + 1)),
            "timestamp": time.time()
        }
        self.save()

    def save(self):
        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)


class BaseVisionClient(ABC):
    """Interfaz base para clientes de modelos multimodales."""

    def analyze_image(self, image_bytes: bytes, mime_type: str = "image/png", custom_prompt: Optional[str] = None) -> str:
        results = self.analyze_images_batch([(image_bytes, mime_type)], custom_prompt=custom_prompt)
        return results[0] if results else ""

    @abstractmethod
    def analyze_images_batch(self, images_list: List[Tuple[bytes, str]], custom_prompt: Optional[str] = None) -> List[str]:
        pass


class GeminiVisionClient(BaseVisionClient):
    """Implementación oficial para Google Gemini SDK con soporte de lotes."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, api_url: Optional[str] = None):
        self.model_name = model_name or MODEL_NAME
        clean_key = api_key if api_key and api_key != "tu_api_key_aqui" else None
        if clean_key:
            self.client = genai.Client(api_key=clean_key)
        else:
            self.client = genai.Client()

    def analyze_images_batch(self, images_list: List[Tuple[bytes, str]], custom_prompt: Optional[str] = None) -> List[str]:
        if not images_list:
            return []

        prompt = custom_prompt or SYSTEM_PROMPT
        parts = []
        for img_bytes, mime_type in images_list:
            parts.append(types.Part.from_bytes(data=img_bytes, mime_type=mime_type))
        
        batch_prompt = prompt + f"\n\nInstrucción especial: Procesa en orden exacto las {len(images_list)} imágenes adjuntas y separa la respuesta de cada página con el delimitador exacto '---PAGINA_NEXT---'."
        parts.append(batch_prompt)

        config = types.GenerateContentConfig(
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )

        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=parts,
                    config=config
                )

                if DELAY_BETWEEN_REQUESTS > 0:
                    time.sleep(DELAY_BETWEEN_REQUESTS)

                text = response.text.strip() if response.text else ""
                blocks = text.split("---PAGINA_NEXT---")
                if len(blocks) == len(images_list):
                    return [b.strip() for b in blocks]
                else:
                    # Fallback si el modelo no usó el delimitador exacto
                    return [text] * len(images_list)

            except errors.APIError as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    if "GenerateRequestsPerDay" in err_msg or "quotaValue" in err_msg or "FreeTier" in err_msg:
                        raise QuotaExhaustedError(
                            f"Se ha alcanzado el límite diario de la cuota Free Tier del modelo '{self.model_name}'."
                        ) from e

                    retry_match = re.search(r'retry in\s+([\d\.]+)s', err_msg, re.IGNORECASE)
                    wait_time = float(retry_match.group(1)) + 2.0 if retry_match else 55.0

                    print(f"\n[WARN] Rate limit 429 (RPM). Esperando {wait_time:.1f}s antes del reintento {attempt + 1}/{max_attempts}...")
                    time.sleep(wait_time)
                else:
                    raise e
            except Exception as e:
                err_msg = str(e)
                if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
                    if "GenerateRequestsPerDay" in err_msg or "quotaValue" in err_msg:
                        raise QuotaExhaustedError(
                            f"Se ha alcanzado el límite diario de la cuota Free Tier del modelo '{self.model_name}'."
                        ) from e
                    wait_time = 55.0
                    print(f"\n[WARN] Rate limit 429 detectado. Esperando {wait_time}s antes de reintentar...")
                    time.sleep(wait_time)
                else:
                    raise e

        raise Exception("Se superaron todos los reintentos tras varios errores 429 de la API.")


class OpenRouterVisionClient(BaseVisionClient):
    """Implementación eficiente para OpenRouter enviando lotes de múltiples imágenes en 1 sola llamada API."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None, api_url: Optional[str] = None):
        self.api_key = api_key or API_KEY
        self.model_name = model_name or MODEL_NAME or "google/gemini-2.5-flash"
        self.session = requests.Session()
        
        base_url = (api_url or API_URL or "https://openrouter.ai/api/v1").rstrip("/")
        if not base_url.endswith("/chat/completions"):
            self.endpoint = f"{base_url}/chat/completions"
        else:
            self.endpoint = base_url

    def analyze_images_batch(self, images_list: List[Tuple[bytes, str]], custom_prompt: Optional[str] = None) -> List[str]:
        if not images_list:
            return []

        prompt = custom_prompt or SYSTEM_PROMPT
        
        # Construir contenido multimodal con todas las imágenes del lote
        user_content = [{"type": "text", "text": prompt + f"\n\nProcesa en orden exacto las {len(images_list)} páginas/imágenes adjuntas en este lote. Separa el resultado de cada página con el delimitador '---PAGINA_NEXT---'."}]
        
        for img_bytes, mime_type in images_list:
            b64_image = base64.b64encode(img_bytes).decode("utf-8")
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64_image}"}
            })

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/libro-edificio",
            "X-Title": "Libro del Edificio Processor"
        }

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": user_content
                }
            ]
        }

        max_attempts = 4
        for attempt in range(max_attempts):
            try:
                resp = self.session.post(self.endpoint, headers=headers, json=payload, timeout=120)
                
                if resp.status_code == 200:
                    data = resp.json()
                    choices = data.get("choices", [])
                    if choices and "message" in choices[0]:
                        content = choices[0]["message"].get("content", "")
                        if DELAY_BETWEEN_REQUESTS > 0:
                            time.sleep(DELAY_BETWEEN_REQUESTS)
                        
                        blocks = content.split("---PAGINA_NEXT---")
                        if len(blocks) == len(images_list):
                            return [b.strip() for b in blocks]
                        else:
                            return [content.strip()] * len(images_list)
                    return [""] * len(images_list)

                if resp.status_code in (429, 402):
                    err_text = resp.text
                    if resp.status_code == 402 or "insufficient_quota" in err_text or "out of credits" in err_text.lower():
                        raise QuotaExhaustedError("Saldo o cuota de OpenRouter insuficiente. Revisa tus créditos en openrouter.ai")

                    wait_time = (attempt + 1) * 15.0
                    print(f"\n[WARN OpenRouter] HTTP {resp.status_code}. Esperando {wait_time}s (reintento {attempt + 1}/{max_attempts})...")
                    time.sleep(wait_time)
                else:
                    raise Exception(f"OpenRouter API Error HTTP {resp.status_code}: {resp.text}")

            except requests.RequestException as e:
                print(f"\n[WARN OpenRouter] Error de conexión: {e}. Reintentando ({attempt + 1}/{max_attempts})...")
                time.sleep(5.0)

        raise Exception(f"Error al comunicar con OpenRouter tras {max_attempts} reintentos.")


def get_vision_client(
    provider: Optional[str] = None,
    api_key: Optional[str] = None,
    model_name: Optional[str] = None,
    api_url: Optional[str] = None
) -> BaseVisionClient:
    """Factoría para instanciar el cliente visual según el PROVEEDOR seleccionado."""
    prov = (provider or PROVEEDOR).upper()
    key = api_key if api_key is not None else API_KEY
    model = model_name or MODEL_NAME
    url = api_url or API_URL

    if prov == "OPENROUTER" or prov in {"OPENAI", "DEEPSEEK", "KIMI", "Z.AI"}:
        return OpenRouterVisionClient(api_key=key, model_name=model, api_url=url)
    elif prov == "GEMINI":
        return GeminiVisionClient(api_key=key, model_name=model, api_url=url)
    else:
        return OpenRouterVisionClient(api_key=key, model_name=model, api_url=url)


class DocumentProcessor:
    """Clase encargada de la extracción híbrida con Fast-Path Local y lotes multipágina."""

    def __init__(self, vision_client: BaseVisionClient, checkpoint_mgr: CheckpointManager):
        self.vision_client = vision_client
        self.checkpoint_mgr = checkpoint_mgr

    def process_file_incremental(
        self,
        file_path: Path,
        rel_path: str,
        target_md_path: Path,
        group_filename: str
    ) -> bool:
        ext = file_path.suffix.lower()

        if ext in IGNORED_EXTENSIONS or ext not in ALLOWED_EXTENSIONS:
            return False

        if ext == ".pdf":
            return self._process_pdf_optimized(file_path, rel_path, target_md_path, group_filename)
        elif ext in {".png", ".jpg", ".jpeg"}:
            return self._process_image_incremental(file_path, rel_path, target_md_path, group_filename)

        return False

    def _process_pdf_optimized(
        self,
        pdf_path: Path,
        rel_path: str,
        target_md_path: Path,
        group_filename: str
    ) -> bool:
        """Procesa un PDF con Fast-Path Local para texto vectorial y lotes multipágina para escaneos."""
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            header_written = False

            pending_api_pages: List[Tuple[int, bytes, str]] = []  # (page_num, img_bytes, mime_type)

            def flush_api_batch():
                nonlocal header_written
                if not pending_api_pages:
                    return

                images_list = [(img, mime) for _, img, mime in pending_api_pages]
                page_numbers = [p_num for p_num, _, _ in pending_api_pages]

                # 1 sola llamada API para todo el lote multipágina
                results_md = self.vision_client.analyze_images_batch(images_list)

                with open(target_md_path, "a", encoding="utf-8") as out_f:
                    for idx, page_num in enumerate(page_numbers):
                        page_md = results_md[idx] if idx < len(results_md) else results_md[0]
                        if not header_written and not self.checkpoint_mgr.data.get(rel_path, {}).get("pages_completed"):
                            out_f.write(f"\n\n# DOCUMENTO: [{rel_path}]\n\n")
                            header_written = True

                        out_f.write(f"### Página {page_num}\n\n{page_md}\n\n")
                        self.checkpoint_mgr.mark_page_completed(rel_path, page_num, group_filename, total_pages, save_disk=False)

                self.checkpoint_mgr.save()
                pending_api_pages.clear()

            for page_idx in range(total_pages):
                page_num = page_idx + 1

                if self.checkpoint_mgr.is_page_processed(rel_path, page_num):
                    continue

                page = doc.load_page(page_idx)
                raw_text = page.get_text("text").strip()
                images = page.get_images()

                # Vía Rápida Local (Fast-Path PyMuPDF) - Coste $0 / Milisegundos
                if ENABLE_FAST_LOCAL_PATH and len(raw_text) >= MIN_TEXT_CHARS:
                    with open(target_md_path, "a", encoding="utf-8") as out_f:
                        if not header_written and not self.checkpoint_mgr.data.get(rel_path, {}).get("pages_completed"):
                            out_f.write(f"\n\n# DOCUMENTO: [{rel_path}]\n\n")
                            header_written = True

                        out_f.write(f"### Página {page_num} (Fast-Path Local)\n\n{raw_text}\n\n")
                        self.checkpoint_mgr.mark_page_completed(rel_path, page_num, group_filename, total_pages, save_disk=False)
                else:
                    # Acumular para llamada API por lote
                    pix = page.get_pixmap(dpi=RENDER_DPI)
                    img_bytes = pix.tobytes("png")
                    pending_api_pages.append((page_num, img_bytes, "image/png"))

                    if len(pending_api_pages) >= PAGES_PER_BATCH:
                        flush_api_batch()

            # Procesar páginas pendientes en el último lote
            flush_api_batch()
            self.checkpoint_mgr.save()

            doc.close()

            if self.checkpoint_mgr.is_file_processed(rel_path):
                with open(target_md_path, "a", encoding="utf-8") as out_f:
                    out_f.write("---\n")

            return True

        except QuotaExhaustedError:
            raise
        except Exception as e:
            print(f"\n[ERROR] Error procesando PDF {rel_path}: {e}")
            return False

    def _process_image_incremental(
        self,
        img_path: Path,
        rel_path: str,
        target_md_path: Path,
        group_filename: str
    ) -> bool:
        if self.checkpoint_mgr.is_file_processed(rel_path):
            return True

        try:
            with Image.open(img_path) as img:
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                img_bytes = buf.getvalue()

            extracted = self.vision_client.analyze_image(img_bytes, mime_type="image/png")
            content_md = f"### Imagen: {img_path.name}\n\n{extracted}"

            with open(target_md_path, "a", encoding="utf-8") as out_f:
                out_f.write(f"\n\n# DOCUMENTO: [{rel_path}]\n\n{content_md}\n\n---\n")

            self.checkpoint_mgr.mark_file_processed(rel_path, group_filename, total_pages=1)
            return True

        except QuotaExhaustedError:
            raise
        except Exception as e:
            print(f"\n[ERROR] Error procesando imagen {rel_path}: {e}")
            return False
