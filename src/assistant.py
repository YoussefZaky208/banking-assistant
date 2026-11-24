"""The assistant: one place that decides what to do with each customer message.
The notebook, the API and the tests all call Assistant.reply(), so they behave the same."""
from src.rag.generator import Generator
from src.rag.knowledge_base import detect_lang
from src.rag.retriever import Retriever
from src.rag.rewriter import QueryRewriter
from src.safety import mask_sensitive
from src.support import TicketStore

MESSAGES = {
    "unclear": {
        "en": "Could you tell me a bit more about what happened? For example: a declined payment, a failed transfer, or a problem with your card.",
        "ar": "هل يمكنك توضيح ما حدث بشكل أكبر؟ مثلًا: عملية دفع مرفوضة، أو تحويل فشل، أو مشكلة في بطاقتك.",
    },
    "off_topic": {
        "en": "I can only help with Papyrus Bank questions, like cards, payments, transfers and fees. What can I help you with?",
        "ar": "أستطيع المساعدة فقط في أسئلة بنك بابيروس، مثل البطاقات والمدفوعات والتحويلات والرسوم. كيف يمكنني مساعدتك؟",
    },
    "sensitive": {
        "en": "For your safety, please never share your PIN, full card number, passcode or SMS codes in a chat, not even with us. I've hidden the numbers you sent.",
        "ar": "حفاظًا على أمانك، لا تشارك أبدًا رقمك السري أو رقم بطاقتك كاملًا أو رمز الدخول أو الرموز التي تصلك برسائل نصية في أي محادثة، حتى معنا. لقد أخفيت الأرقام التي أرسلتها.",
    },
    "injection": {
        "en": "I can't change my rules or how I work, but I'm happy to give you the accurate information:",
        "ar": "لا يمكنني تغيير قواعدي أو طريقة عملي، لكن يسعدني أن أقدم لك المعلومات الصحيحة:",
    },
    "injection_only": {
        "en": "I can't change my rules or how I work. How can I help you with your banking today?",
        "ar": "لا يمكنني تغيير قواعدي أو طريقة عملي. كيف يمكنني مساعدتك في أمورك البنكية اليوم؟",
    },
    "handoff": {
        "en": "I've passed your request to our support team (ticket {ticket}). An agent will continue this conversation and will see a summary, so you won't need to repeat yourself. (Demo: no real agent is connected.)",
        "ar": "لقد حوّلت طلبك إلى فريق الدعم (رقم التذكرة {ticket}). سيكمل أحد الموظفين هذه المحادثة وسيطّلع على ملخص لها، فلن تحتاج إلى تكرار ما قلته. (نسخة تجريبية: لا يوجد موظف حقيقي متصل.)",
    },
}


class Assistant:
    def __init__(self, retriever: Retriever, generator: Generator, rewriter: QueryRewriter,
                 tickets: TicketStore, top_k: int = 4):
        self.retriever = retriever
        self.generator = generator
        self.rewriter = rewriter
        self.tickets = tickets
        self.top_k = top_k

    def reply(self, message: str, history: list | None = None) -> dict:
        history = history or []
        lang = detect_lang(message)                          # language: decided in code

        message, had_secret = mask_sensitive(message)        # secrets never reach the LLM
        warning = MESSAGES["sensitive"][lang] + "\n\n" if had_secret else ""

        analysis = self.rewriter.analyze(message, history)   # route + search query + mood + injection
        route = analysis["route"]
        sources, source_ids, ticket = [], [], None

        if analysis["injection"] and route != "question":   # manipulation with no real question
            route = "injection"
            text = MESSAGES["injection_only"][lang]
        elif route == "chitchat":
            text = self.generator.small_talk(message, history)
        elif route in ("unclear", "off_topic"):
            text = MESSAGES[route][lang]
        elif route == "handoff":
            conversation = history + [{"role": "user", "content": message}]
            ticket = self.tickets.create(self.generator.summarize(conversation), lang)
            text = MESSAGES["handoff"][lang].format(ticket=ticket["id"])
        else:                                                # a real banking question: RAG
            chunks = self.retriever.search(analysis["query"], top_k=self.top_k, mode="dense", lang=lang)
            result = self.generator.answer(analysis["query"], chunks, lang=lang, upset=analysis["upset"])
            text, sources, source_ids = result["answer"], result["sources"], result["source_ids"]
            if analysis["injection"]:                         # say no to the trick, then answer
                text = MESSAGES["injection"][lang] + "\n\n" + text

        return {"reply": warning + text, "route": route, "query": analysis["query"],
                "upset": analysis["upset"], "injection": analysis["injection"],
                "sources": sources, "source_ids": source_ids, "ticket": ticket,
                "masked_message": message}
