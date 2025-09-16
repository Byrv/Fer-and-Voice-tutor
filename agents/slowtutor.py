from rag import RAGMemory
from langchain_aws.chat_models import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Optional
import time 
from appstate import AppState
import threading
import logging
from polly import PollySpeaker 
from langchain.memory import ConversationBufferMemory

logging.basicConfig(
    level=logging.INFO,
    format="🧠 [%(asctime)s] %(message)s",
    datefmt="%H:%M:%S"
)
voice = PollySpeaker(region="us-east-1", 
                     voice="Kajal", 
                     engine="neural", 
                     lang="en-IN")

memory = ConversationBufferMemory(return_messages=True)
def build_system_prompt(language_instruction: str, age_instruction: str, slow_mode: bool) -> str:
    # Core tutor persona
    base = f"""
You are a kind, patient tutor for kids.

{language_instruction}

{age_instruction}

General rules:
- Be positive and encouraging. Avoid scary or sensitive topics.
- Ask at most one short clarifying question if the problem is ambiguous; otherwise proceed.
- Prefer short sentences and bullets over long paragraphs.
- Never include private or personal data.
- Keep math and logic correct; if unsure, say so.

Output format for full solutions (Markdown):
### Let's Solve It
(One-line friendly intro)

### Steps
1. ...
2. ...
3. ...

### Final Answer
**...**

### Quick Check
(A tiny check to verify the answer)

### Try This
(One similar practice problem with no solution)
"""
    if not slow_mode:
        return base + "\nSlow mode: OFF."

    # Additional behavior for slow thinking mode
    slow = """
Slow Thinking Mode (Socratic + hints):
- We proceed in phases: HINT_1 (general) → HINT_2 (more specific) → HINT_3 (first step) → SOLUTION.
- In HINT phases:
  - Give only ONE short hint (≤ 30 words).
  - Ask one short question to the child like: “What do you think we should do next?”
  - DO NOT compute final values. DO NOT reveal the answer.
- In SOLUTION phase:
  - Provide the full structured solution using the format above.
  - If the child attempted an answer, first evaluate it gently, then correct or confirm.
"""
    return base + slow


def format_context_block(context: Optional[str]) -> str:
    if not context:
        return ""
    return f"**Use these notes if helpful (context):**\n{context}\n\n"

def human_prompt_clarify(question: str, context_block: str) -> str:
    return (
        f"{context_block}"
        f"Student's question:\n{question}\n\n"
        "PHASE: CLARIFY\n"
        "Task: Ask exactly ONE short clarifying question if needed. "
        "Keep it simple. Do not give hints or steps yet."
    )

def human_prompt_hint(question: str, context_block: str, level: int, child_reply: Optional[str]) -> str:
    child_snippet = f"\nChild said: {child_reply}\n" if child_reply else ""
    specificity = {
        1: "Give a general nudge that points to the idea needed.",
        2: "Be more specific, pointing to the operation or relation needed.",
        3: "Reveal the first small step, but not the final answer."
    }[level]
    return (
        f"{context_block}"
        f"Student's question:\n{question}\n"
        f"{child_snippet}\n"
        f"PHASE: HINT_{level}\n"
        f"{specificity}\n"
        "Constraints:\n"
        "- ONE hint only (≤ 30 words).\n"
        "- Ask one short follow-up question.\n"
        "- Do NOT compute final numbers. Do NOT reveal the final answer."
    )

def normalize_choice(text: str) -> str:
    """
    Map user input to normalized choices:
    - 'another hint' → 'hint'
    - 'first step' / 'step' → 'step'
    - 'answer' / 'solution' → 'answer'
    - 'stop' / 'quit' / 'exit' → 'stop'
    - anything else → 'free'
    """
    if not text:
        return "free"
    t = text.strip().lower()
    if any(k in t for k in ["stop", "quit", "exit", "cancel"]):
        return "stop"
    if any(k in t for k in ["answer", "solution", "show answer", "final"]):
        return "answer"
    if any(k in t for k in ["step", "first step", "show step"]):
        return "step"
    if any(k in t for k in ["hint", "another hint", "more hint", "next hint"]):
        return "hint"
    return "free"

def human_prompt_solution(question: str, context_block: str, child_reply: Optional[str]) -> str:
    attempt = f"\nChild attempted: {child_reply}\n" if child_reply else ""
    return (
        f"{context_block}"
        f"Student's question:\n{question}\n"
        f"{attempt}\n"
        "PHASE: SOLUTION\n"
        "Task: Provide the full solution in the required Markdown format. "
        "If the child's attempt is correct, praise and confirm briefly first; "
        "if not, gently correct, then show the solution."
    )

def confusion_prompt(raw_question: str, context_block: str) -> str:
    return (
        f"{context_block}"
        f"Student seems confused while solving:\n{raw_question}\n\n"
        "PHASE: EMOTION_SUPPORT\n"
        "Task: Gently acknowledge the confusion and offer encouragement. "
        "Ask if they'd like a hint or help. Keep it short, friendly, and supportive.\n"
        "Do NOT solve the problem yet. Do NOT give steps or answers."
    )

def emotion_monitor(state: AppState, llm: ChatBedrock, context_block: str, raw_question: str):
    while True:
        if state.interrupt_flag:
            emotion = state.emotion_state
            now = time.time()

            if emotion == "confused":
                # 🔥 For testing, respond immediately (remove cooldown for now)
                # if you want the 30s delay, put the if-condition back
                print("🤖 Emotion Monitor: Detected confusion. Responding...", flush=True)

                support_msg = HumanMessage(content=confusion_prompt(raw_question, context_block))

                try:
                    resp = llm.invoke([support_msg])
                    text = getattr(resp, "content", None) or str(resp)
                    print(f"\n🧠 Tutor (emotion response): {text}\n", flush=True)

                except Exception as e:
                    print(f"⚠️ Emotion Monitor Error: {e}", flush=True)

                state.clear_interrupt()

            elif emotion == "happy":
                print("🎉 Emotion Monitor: You're doing great! Keep it up!", flush=True)
                state.clear_interrupt()

            elif emotion == "frustrated":
                print("💡 Emotion Monitor: Let's slow down, take a deep breath, and try again.", flush=True)
                state.clear_interrupt()

        # 🔄 Small sleep to avoid CPU spin
        time.sleep(1)



class SlowTutor:
    def __init__(self, llm, language_instruction, age_instruction, state, slow_mode=True, max_hints=3):
        self.llm = llm
        self.language_instruction = language_instruction
        self.age_instruction = age_instruction
        self.slow_mode = slow_mode
        self.max_hints = max_hints
        self.state = state
        self.memory = ConversationBufferMemory(return_messages=True)
        self.system_msg = SystemMessage(content=build_system_prompt(language_instruction, age_instruction, slow_mode))

    def _invoke(self, messages):
        return self.llm.invoke(messages)

    
    def handle_interrupt(self):
        emotion = self.state.emotion_state
        print(f"\n🚨 INTERRUPT TRIGGERED: Emotion = {emotion}")
        # ✅ Customize this logic as needed
        if "confused" in emotion:
            print("🧠 Let's slow down and explain things more clearly.")
        elif "frustrated" in emotion:
            print("💡 Let's take a break or try a simpler problem.")
        elif "happy" in emotion:
            print("🎉 Great job! Let's keep going.")
        else:
            print("⚡ Handling generic emotion state.")
        self.state.clear_interrupt()
        print("🔄 Resuming normal tutor flow...\n")


    def tutor_once(self, raw_question: str):
        # Retrieve optional context
        # fer is called 
        # we check if the fer vairable is set to true or false we put this code inside the while loop and check it the varaible state is set to true we need 
        # a state management system
        # if the state changes we read a dynamic varaibel which keep track of the emotion once the viable is set value changes we pause the prgram execution and then 
        # execute diffrent instructions and then we return back to the original prgram flow  
        

        history = self.memory.load_memory_variables({}).get("history", [])  # fresh run per problem
        messages = [self.system_msg] + history + [HumanMessage(content=raw_question)]
        # ✅ Start emotion monitor thread
        monitor_thread = threading.Thread(
            target=emotion_monitor,
            args=(self.state, self.llm, context_block, raw_question),
            daemon=True
        )
        monitor_thread.start()

        # Optional: quick ambiguity check (one clarifying question)
        print("\n🤖 Tutor (slow mode):\n")
        clarify_msg = HumanMessage(content=human_prompt_clarify(raw_question, context_block))
        resp = self._invoke(messages)
        answer_text = _to_text(resp)


        # Get child input to the clarifying question (optional)
        child_clarify = input("\n🧒 Your turn (answer or press Enter to skip): ").strip()
        # Append the AI and child's reply into history for context
        history.extend([clarify_msg, AIMessage(content=resp.content)])
        if child_clarify:
            history.append(HumanMessage(content=f"Child's clarification: {child_clarify}"))

        # Hints loop (if slow mode)
        child_reply: Optional[str] = None
        if self.slow_mode:
            level = 1

        while level <= self.max_hints:
           
            # Proceed with hint generation
            hint_msg = HumanMessage(content=human_prompt_hint(raw_question, context_block, level, child_reply))
            resp = self._invoke(history + [hint_msg])
            child_reply = input("\n🧒 Your turn (type 'another hint', 'first step', 'answer', or your idea): ").strip()
            choice = normalize_choice(child_reply)

            history.extend([hint_msg, AIMessage(content=resp.content)])
            if child_reply:
                history.append(HumanMessage(content=f"Child response: {child_reply}"))

            if choice == "answer":
                break
            elif choice == "step":
                level = max(level, 3)
            elif choice in ("hint", "free"):
                level += 1
            elif choice == "stop":
                print("\n✨ No problem. We can stop here. Come back anytime!")
                return
            else:
                level += 1


        # Final solution phase
        solution_msg = HumanMessage(content=human_prompt_solution(raw_question, context_block, child_reply))
        resp = self._invoke(history + [solution_msg])
        # Store in memory nicely
        full_text = f"Q: {raw_question}\n\nA:\n{resp.content}"
        
        self.memory.chat_memory.add_user_message(raw_question)
        self.memory.chat_memory.add_ai_message(answer_text)

        return answer_text

