"""
mock_interview.py
-----------------
Simulador interactivo de entrevista técnica usando Google Gemini.
El entrevistador actúa como reclutador senior y hace preguntas contextuales
basadas en el CV del candidato y la descripción del trabajo (JD).
"""

import os
from datetime import datetime
from typing import Optional

from google import genai
from google.genai import types


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
MAX_QUESTIONS = 7
OUTPUT_TRANSCRIPT_PATH = "output/interview_transcript.md"


# ---------------------------------------------------------------------------
# System prompt del entrevistador
# ---------------------------------------------------------------------------
INTERVIEWER_SYSTEM_PROMPT = """
Eres un reclutador técnico senior en una empresa de tecnología. 
Tu trabajo es evaluar a candidatos junior para posiciones de desarrollo de software.

REGLAS ESTRICTAS que debes seguir siempre:
1. SOLO haz preguntas sobre tecnologías, habilidades y experiencias que aparecen 
   explícitamente en el CV del candidato o en la descripción del trabajo (JD).
2. NUNCA preguntes sobre tecnologías que no estén en el CV ni en el JD.
3. Haz UNA sola pregunta por turno, espera la respuesta antes de continuar.
4. Adapta la dificultad al nivel junior/trainee del candidato.
5. Sé profesional pero amigable para reducir el estrés del candidato.
6. Cuando hayas hecho {max_questions} preguntas técnicas, cierra la entrevista 
   con un feedback constructivo y honesto sobre el desempeño del candidato.
7. No salgas del contexto de la entrevista. Si el usuario habla de otro tema, 
   redirigelo amablemente a la entrevista.
8. El feedback final debe mencionar: fortalezas observadas, áreas de mejora 
   y una recomendación general.

Formato de tu primera respuesta:
- Saluda al candidato por su nombre
- Preséntate brevemente
- Explica el formato de la entrevista (cuántas preguntas)
- Haz la primera pregunta técnica

Recuerda: eres un evaluador justo, no intimidante. Tu objetivo es descubrir 
el potencial real del candidato.
""".strip()


class MockInterviewService:
    """
    Servicio de simulación de entrevista técnica con memoria de conversación.

    Attributes:
        profile (dict): Perfil del candidato cargado desde el YAML.
        job_description (str): Descripción de la vacante.
        api_key (str): API key de Google Gemini.
        messages (list): Historial de mensajes de la conversación.
        transcript (list): Registro de la transcripción para exportar.
        question_count (int): Contador de preguntas realizadas.
        interview_finished (bool): Indica si la entrevista ha concluido.
    """

    def __init__(self, profile: dict, job_description: str, api_key: Optional[str] = None) -> None:
        """
        Inicializa el servicio de entrevista.

        Args:
            profile (dict): Datos del candidato (nombre, skills, experiencia, etc.).
            job_description (str): Texto completo de la descripción del trabajo.
            api_key (str, optional): API key de Gemini. Si no se pasa, la lee de 
                                     la variable de entorno GEMINI_API_KEY.

        Raises:
            ValueError: Si no se encuentra una API key válida.
        """
        self.profile = profile
        self.job_description = job_description
        self.messages: list = []
        self.transcript: list = []
        self.question_count: int = 0
        self.interview_finished: bool = False

        # Configurar API key
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

    def _build_system_prompt(self) -> str:
        """
        Construye el system prompt inyectando el CV y el JD como contexto.

        Returns:
            str: System prompt completo con contexto del candidato y la vacante.
        """
        candidate_name = self.profile.get("name", "el candidato")
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

        cv_context = f"""
=== CV DEL CANDIDATO ===
Nombre: {candidate_name}
Posición buscada: {self.profile.get('target_position', 'Desarrollador Junior')}
Email: {self.profile.get('email', 'N/A')}

Skills técnicos: {', '.join(skills) if skills else 'No especificados'}

Experiencia laboral:
{experience_text if experience_text else '  - Sin experiencia laboral previa'}

Educación:
{education_text if education_text else '  - No especificada'}

=== DESCRIPCIÓN DE LA VACANTE (JD) ===
{self.job_description}
========================
"""
        base_prompt = INTERVIEWER_SYSTEM_PROMPT.format(max_questions=MAX_QUESTIONS)
        return base_prompt + "\n\n" + cv_context

    def _call_gemini(self) -> str:
        """
        Llama a la API de Gemini con el historial de mensajes actual.

        Returns:
            str: Respuesta de texto del modelo.
        """
        system_prompt = self._build_system_prompt()

        response = self.client.models.generate_content(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
            ),
            contents=self.messages
        )
        return response.text.strip()

    def _add_to_history(self, role: str, text: str) -> None:
        """
        Agrega un mensaje al historial de conversación y a la transcripción.

        Args:
            role (str): 'user' o 'model'.
            text (str): Contenido del mensaje.
        """
        self.messages.append(
            types.Content(
                role=role,
                parts=[types.Part(text=text)]
            )
        )
        label = "🤵 Entrevistador" if role == "model" else "👤 Candidato"
        self.transcript.append({
            "role": label,
            "text": text,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })

    def _detect_question_in_response(self, text: str) -> bool:
        """
        Detecta si la respuesta del modelo contiene una pregunta técnica nueva.

        Args:
            text (str): Texto de la respuesta del modelo.

        Returns:
            bool: True si contiene una pregunta, False si es feedback final.
        """
        return "?" in text

    # ------------------------------------------------------------------
    # Métodos públicos
    # ------------------------------------------------------------------

    def start_interview(self) -> str:
        """
        Inicia la entrevista enviando el primer mensaje al modelo.

        Returns:
            str: Saludo y primera pregunta del entrevistador.
        """
        candidate_name = self.profile.get("name", "candidato")
        initial_message = (
            f"Hola, soy {candidate_name} y estoy listo/a para comenzar la entrevista."
        )
        self._add_to_history("user", initial_message)
        response = self._call_gemini()
        self._add_to_history("model", response)

        if self._detect_question_in_response(response):
            self.question_count += 1

        return response

    def answer_question(self, user_answer: str) -> str:
        """
        Procesa la respuesta del candidato y obtiene la siguiente pregunta o feedback.

        Args:
            user_answer (str): Respuesta del candidato a la pregunta anterior.

        Returns:
            str: Siguiente pregunta o feedback final del entrevistador.

        Raises:
            RuntimeError: Si se llama después de que la entrevista ya terminó.
        """
        if self.interview_finished:
            raise RuntimeError("La entrevista ya ha concluido. Usa export_transcript() para guardar.")

        self._add_to_history("user", user_answer)
        response = self._call_gemini()
        self._add_to_history("model", response)

        if self._detect_question_in_response(response) and self.question_count < MAX_QUESTIONS:
            self.question_count += 1
        else:
            self.interview_finished = True

        return response

    def is_finished(self) -> bool:
        """
        Indica si la entrevista ha concluido.

        Returns:
            bool: True si la entrevista terminó.
        """
        return self.interview_finished or self.question_count >= MAX_QUESTIONS

    def run_interactive(self) -> None:
        """
        Ejecuta la entrevista de forma interactiva en la terminal.
        """
        print("\n" + "=" * 60)
        print("        🎯 SIMULADOR DE ENTREVISTA TÉCNICA               ")
        print("=" * 60)
        print(f"Candidato : {self.profile.get('name', 'N/A')}")
        print(f"Posición  : {self.profile.get('target_position', 'N/A')}")
        print(f"Preguntas : máximo {MAX_QUESTIONS}")
        print("=" * 60)
        print("💡 Escribe 'salir' en cualquier momento para terminar.\n")

        print("Iniciando entrevista...\n")
        opening = self.start_interview()
        print(f"🤵 Entrevistador:\n{opening}\n")

        while not self.is_finished():
            try:
                user_input = input("👤 Tu respuesta: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n[INFO] Entrevista interrumpida por el usuario.")
                break

            if not user_input:
                print("⚠️  Por favor escribe una respuesta antes de continuar.\n")
                continue

            if user_input.lower() == "salir":
                print("\n[INFO] Saliendo de la entrevista...")
                break

            print("\nProcesando respuesta...\n")
            response = self.answer_question(user_input)
            print(f"🤵 Entrevistador:\n{response}\n")

            if self.is_finished():
                print("=" * 60)
                print("✅ Entrevista finalizada.")
                print("=" * 60)
                break

        self.export_transcript()

    def export_transcript(self, output_path: str = OUTPUT_TRANSCRIPT_PATH) -> None:
        """
        Exporta la transcripción completa de la entrevista a un archivo Markdown.

        Args:
            output_path (str): Ruta del archivo de salida.
        """
        if not self.transcript:
            print("[WARN] No hay transcripción para exportar.")
            return

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        candidate_name = self.profile.get("name", "Candidato")
        position = self.profile.get("target_position", "N/A")
        date_str = datetime.now().strftime("%d/%m/%Y %H:%M")

        lines = [
            "# 📋 Transcripción de Entrevista Técnica\n",
            f"**Candidato:** {candidate_name}  ",
            f"**Posición:** {position}  ",
            f"**Fecha:** {date_str}  ",
            f"**Total de preguntas:** {self.question_count}/{MAX_QUESTIONS}  ",
            "\n---\n",
        ]

        for i, entry in enumerate(self.transcript, start=1):
            role = entry["role"]
            text = entry["text"]
            timestamp = entry["timestamp"]
            lines.append(f"### {i}. {role} _{timestamp}_\n")
            lines.append(f"{text}\n")
            lines.append("---\n")

        lines.append(f"\n_Transcripción generada automáticamente el {date_str}_\n")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"\n[INFO] ✅ Transcripción guardada en: {output_path}")