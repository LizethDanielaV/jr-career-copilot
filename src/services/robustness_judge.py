"""
robustness_judge.py
-------------------
Validador LLM-as-a-Judge que detecta alucinaciones, inconsistencias
y violaciones éticas en el CV generado por el optimizador.
"""

import os
import json
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OUTPUT_REPORT_PATH = "output/robustness_report.json"


# ---------------------------------------------------------------------------
# Modelos Pydantic (Structured Output)
# ---------------------------------------------------------------------------

class Alucinacion(BaseModel):
    """Representa una alucinación o inconsistencia detectada en el CV."""
    linea_cv: str = Field(description="Fragmento exacto del CV donde se detectó el problema")
    dato_inventado: str = Field(description="Descripción del dato falso, exagerado o inconsistente")
    severidad: str = Field(description="Nivel de severidad: baja, media, alta o critica")
    tipo: str = Field(description="Tipo de problema: alucinacion, inconsistencia, exageracion, violacion_etica")
    sugerencia: str = Field(description="Recomendación concreta para corregir el problema")


class ViolacionEtica(BaseModel):
    """Representa una violación ética detectada en el CV."""
    descripcion: str = Field(description="Descripción de la violación ética encontrada")
    fragmento_cv: str = Field(description="Fragmento del CV donde aparece la violación")
    recomendacion: str = Field(description="Cómo corregir la violación ética")


class ReporteRobustez(BaseModel):
    """Reporte completo de validación de robustez del CV."""
    score_honestidad: int = Field(description="Puntuación de honestidad del CV del 0 al 100", ge=0, le=100)
    nivel_confianza: str = Field(description="Nivel de confianza del auditor: bajo, medio, alto")
    alucinaciones_detectadas: list[Alucinacion] = Field(description="Lista de alucinaciones encontradas")
    violaciones_eticas: list[ViolacionEtica] = Field(description="Lista de violaciones éticas detectadas")
    fortalezas_cv: list[str] = Field(description="Lista de aspectos positivos y honestos del CV")
    areas_riesgo: list[str] = Field(description="Áreas del CV que representan riesgo")
    comentario_auditor: str = Field(description="Resumen ejecutivo del auditor")
    recomendacion_final: str = Field(description="Veredicto: APROBADO, APROBADO_CON_OBSERVACIONES o RECHAZADO")
    timestamp: str = Field(description="Fecha y hora de la validación en formato ISO")


# ---------------------------------------------------------------------------
# System prompt del juez
# ---------------------------------------------------------------------------
JUDGE_SYSTEM_PROMPT = """
Eres un auditor experto en verificación de currículums vitae (CV) para empresas de tecnología.
Tu rol es detectar con precisión: alucinaciones, inconsistencias, exageraciones y 
violaciones éticas en CVs generados por IA.

DEFINICIONES:
- Alucinación: Dato completamente inventado que no puede inferirse del perfil original.
- Inconsistencia: Contradicción interna entre secciones del CV.
- Exageración: Habilidad o logro inflado más allá de lo razonable para el nivel del candidato.
- Violación ética: Contenido discriminatorio, engañoso o que podría causar daño.

TU PROCESO DE AUDITORÍA:
1. Compara el CV generado con el perfil original del candidato.
2. Identifica cualquier información en el CV que NO esté respaldada por el perfil original.
3. Detecta exageraciones en skills.
4. Verifica coherencia entre fechas, roles y descripciones.
5. Evalúa si hay contenido éticamente cuestionable.
6. Asigna un score_honestidad del 0 al 100.
7. Emite un veredicto: APROBADO (score >= 80), APROBADO_CON_OBSERVACIONES (50-79), RECHAZADO (<50).

IMPORTANTE: Responde ÚNICAMENTE con el JSON estructurado solicitado, sin texto adicional.
""".strip()


class RobustnessJudgeService:
    """
    Servicio de validación de robustez del CV usando LLM-as-a-Judge.

    Attributes:
        profile (dict): Perfil original del candidato.
        job_description (str): Descripción de la vacante.
        generated_cv (str): CV generado por el optimizador.
        client: Cliente de Google GenAI.
        report (ReporteRobustez | None): Reporte generado tras run_validation().
    """

    def __init__(
        self,
        profile: dict,
        job_description: str,
        generated_cv: str,
        api_key: Optional[str] = None
    ) -> None:
        """
        Inicializa el servicio de validación de robustez.

        Args:
            profile (dict): Perfil original del candidato desde el YAML.
            job_description (str): Texto completo de la descripción del trabajo.
            generated_cv (str): Contenido del CV generado por el optimizador.
            api_key (str, optional): API key de Gemini.

        Raises:
            ValueError: Si no se encuentra una API key válida.
        """
        self.profile = profile
        self.job_description = job_description
        self.generated_cv = generated_cv
        self.report: Optional[ReporteRobustez] = None

        resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No se encontró API key de Gemini. "
                "Define la variable de entorno GEMINI_API_KEY o pásala como argumento."
            )
        self.client = genai.Client(api_key=resolved_key)

    # ------------------------------------------------------------------
    # Métodos privados
    # ------------------------------------------------------------------

    def _build_audit_prompt(self) -> str:
        """Construye el prompt de auditoría con el perfil original y el CV generado."""
        skills = self.profile.get("skills", [])
        experience = self.profile.get("experience", [])
        education = self.profile.get("education", [])

        experience_text = ""
        for exp in experience:
            experience_text += (
                f"  - {exp.get('title', 'N/A')} en {exp.get('company', 'N/A')} "
                f"({exp.get('start_year', '')} - {exp.get('end_year', 'Presente')}): "
                f"{exp.get('description', '')}\n"
            )

        education_text = ""
        for edu in education:
            education_text += (
                f"  - {edu.get('degree', 'N/A')} en {edu.get('institution', 'N/A')} "
                f"({edu.get('year', 'N/A')})\n"
            )

        return f"""
Audita el siguiente CV generado por IA comparándolo con el perfil original del candidato.

=== PERFIL ORIGINAL DEL CANDIDATO (FUENTE DE VERDAD) ===
Nombre: {self.profile.get('name', 'N/A')}
Posición buscada: {self.profile.get('target_position', 'N/A')}
Skills declarados: {', '.join(skills) if skills else 'Ninguno'}

Experiencia laboral real:
{experience_text if experience_text else '  - Sin experiencia laboral previa'}

Educación real:
{education_text if education_text else '  - No especificada'}

=== DESCRIPCIÓN DE LA VACANTE (JD) ===
{self.job_description[:1000]}

=== CV GENERADO POR IA (A AUDITAR) ===
{self.generated_cv}

=== INSTRUCCIONES ===
Compara cada sección del CV generado contra el perfil original.
Marca como alucinación TODO dato en el CV que no esté en el perfil original.
El timestamp debe ser: {datetime.now().isoformat()}
""".strip()

    def _call_gemini_structured(self) -> dict:
        """
        Llama a Gemini con Structured Outputs para obtener un JSON validado.

        Returns:
            dict: Respuesta JSON parseada del modelo.
        """
        schema = ReporteRobustez.model_json_schema()
        prompt = self._build_audit_prompt()

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=JUDGE_SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=schema,
            ),
            contents=prompt
        )

        raw_text = response.text.strip()

        # Limpiar posibles backticks de markdown
        if raw_text.startswith("```"):
            raw_text = raw_text.split("```")[1]
            if raw_text.startswith("json"):
                raw_text = raw_text[4:]
            raw_text = raw_text.strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError as e:
            raise ValueError(f"La respuesta de Gemini no es JSON válido: {e}\nRespuesta: {raw_text[:500]}")

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def run_validation(self) -> ReporteRobustez:
        """
        Ejecuta la validación completa del CV y genera el reporte de robustez.

        Returns:
            ReporteRobustez: Reporte completo con score, alucinaciones y veredicto.
        """
        print("\n" + "=" * 60)
        print("     🔍 VALIDADOR DE ROBUSTEZ — LLM-as-a-Judge           ")
        print("=" * 60)
        print(f"[INFO] Auditando CV de: {self.profile.get('name', 'N/A')}")
        print("[INFO] Enviando CV al juez (Gemini)... esto puede tomar unos segundos.\n")

        json_data = self._call_gemini_structured()

        if "timestamp" not in json_data or not json_data["timestamp"]:
            json_data["timestamp"] = datetime.now().isoformat()

        self.report = ReporteRobustez(**json_data)
        self._print_summary()
        self.export_report()

        return self.report

    def _print_summary(self) -> None:
        """Imprime un resumen legible del reporte en la consola."""
        if not self.report:
            return

        r = self.report
        print("=" * 60)
        print("📊 RESUMEN DEL REPORTE DE ROBUSTEZ")
        print("=" * 60)
        print(f"  Score de honestidad : {r.score_honestidad}/100")
        print(f"  Nivel de confianza  : {r.nivel_confianza}")
        print(f"  Veredicto final     : {r.recomendacion_final}")
        print(f"  Alucinaciones       : {len(r.alucinaciones_detectadas)} detectadas")
        print(f"  Violaciones éticas  : {len(r.violaciones_eticas)} detectadas")
        print(f"  Fortalezas del CV   : {len(r.fortalezas_cv)} identificadas")
        print("-" * 60)
        print(f"💬 Comentario del auditor:\n{r.comentario_auditor}")

        if r.alucinaciones_detectadas:
            print("\n⚠️  ALUCINACIONES DETECTADAS:")
            for i, a in enumerate(r.alucinaciones_detectadas, 1):
                print(f"  [{i}] Severidad: {a.severidad.upper()}")
                print(f"      Fragmento : \"{a.linea_cv[:80]}\"")
                print(f"      Problema  : {a.dato_inventado}")
                print(f"      Sugerencia: {a.sugerencia}\n")

        if r.violaciones_eticas:
            print("🚨 VIOLACIONES ÉTICAS:")
            for i, v in enumerate(r.violaciones_eticas, 1):
                print(f"  [{i}] {v.descripcion}")
                print(f"      Recomendación: {v.recomendacion}\n")

        print("=" * 60)

    def export_report(self, output_path: str = OUTPUT_REPORT_PATH) -> None:
        """
        Exporta el reporte de robustez a un archivo JSON válido.

        Args:
            output_path (str): Ruta del archivo de salida.

        Raises:
            RuntimeError: Si se llama antes de run_validation().
        """
        if not self.report:
            raise RuntimeError(
                "No hay reporte para exportar. Ejecuta run_validation() primero."
            )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        report_dict = self.report.model_dump()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2)

        print(f"\n[INFO] ✅ Reporte de robustez guardado en: {output_path}")