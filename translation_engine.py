from langdetect import DetectorFactory, LangDetectException, detect_langs
from googletrans import Translator

# Make detection deterministic across runs.
DetectorFactory.seed = 0


LANGUAGE_NAME_BY_CODE = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "pa": "Punjabi",
    "ta": "Tamil",
    "kn": "Kannada",
    "te": "Telugu",
    "ml": "Malayalam",
    "ar": "Arabic",
    "pt": "Portuguese",
    "es": "Spanish",
    "fr": "French",
    "ur": "Urdu",
}


class DocumentTranslationEngine:
    """Language detection + translation utility for OCR outputs."""

    def __init__(self):
        self.translator = Translator()

    def detect_language(self, text: str):
        """Return detected language code, language name, and confidence."""
        content = (text or "").strip()
        if len(content) < 3:
            return "en", "English", 0.0

        try:
            candidates = detect_langs(content)
            if not candidates:
                return "en", "English", 0.0
            top = candidates[0]
            code = top.lang
            confidence = float(getattr(top, "prob", 0.0))
            name = LANGUAGE_NAME_BY_CODE.get(code, code.upper())
            return code, name, confidence
        except LangDetectException:
            return "en", "English", 0.0
        except Exception:
            return "en", "English", 0.0

    def _translate_chunks(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """Translate long text safely by chunking to avoid API size issues."""
        content = text or ""
        if not content.strip():
            return content

        max_chunk = 3500
        chunks = []
        current = []
        current_len = 0

        for line in content.split("\n"):
            line_len = len(line) + 1
            if current and current_len + line_len > max_chunk:
                chunks.append("\n".join(current))
                current = [line]
                current_len = line_len
            else:
                current.append(line)
                current_len += line_len

        if current:
            chunks.append("\n".join(current))

        translated_parts = []
        for chunk in chunks:
            try:
                result = self.translator.translate(chunk, src=source_language, dest=target_language)
                translated_parts.append(result.text)
            except Exception:
                translated_parts.append(chunk)

        return "\n".join(translated_parts)

    def translate_text(self, text: str, target_language: str, source_language: str = "auto") -> str:
        """Translate text to target language; on failure return original text."""
        if target_language in ("", "original"):
            return text
        return self._translate_chunks(text, target_language, source_language)

    def translate_data(self, data: dict, target_language: str, source_language: str = "auto") -> dict:
        """Translate OCR structured payload while preserving table schema."""
        if target_language in ("", "original"):
            return data

        translated = {
            "kv": {},
            "table": [],
            "full_text": self.translate_text(str(data.get("full_text", "") or ""), target_language, source_language),
        }

        for key, value in (data.get("kv", {}) or {}).items():
            value_text = str(value) if value is not None else ""
            translated["kv"][key] = self.translate_text(value_text, target_language, source_language)

        for row in (data.get("table", []) or []):
            if isinstance(row, dict):
                row_out = {}
                for col, value in row.items():
                    if isinstance(value, str):
                        row_out[col] = self.translate_text(value, target_language, source_language)
                    else:
                        row_out[col] = value
                translated["table"].append(row_out)
            else:
                translated["table"].append(row)

        return translated
