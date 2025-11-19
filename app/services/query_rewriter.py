# app/services/query_rewriter.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class QueryRewriter:
    """
    Viết lại câu hỏi của người dùng dựa trên ngữ cảnh lịch sử chat
    để tạo query tốt hơn cho vector search
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=0.1  # Giảm temperature để output nhất quán
        )
        
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """Bạn là chuyên gia viết lại câu hỏi (Query Rewriter).

**Nhiệm vụ:** 
Viết lại câu hỏi của người dùng thành câu hỏi ĐỘC LẬP, RÕ RÀNG, PHÙ HỢP cho tìm kiếm vector.

**Nguyên tắc:**
1. Nếu câu hỏi hiện tại CÓ đại từ nhân xưng/chỉ định mơ hồ (như "họ", "đó", "này", "thế", "vậy",...) → Thay thế bằng thông tin CỤ THỂ từ lịch sử
2. Nếu câu hỏi thiếu ngữ cảnh → Bổ sung thông tin từ lịch sử
3. Nếu câu hỏi đã RÕ RÀNG và ĐỘC LẬP → Giữ nguyên hoặc chỉnh sửa NHẸ
4. Luôn giữ ý định gốc của người dùng
5. Câu viết lại phải TỰ NHIÊN, NGẮN GỌN (1-2 câu)

**Ví dụ:**

Lịch sử: "Quy định về khóa luận tốt nghiệp?"
Câu hỏi: "Thời hạn nộp?"
→ Viết lại: "Thời hạn nộp khóa luận tốt nghiệp là khi nào?"

Lịch sử: không có
Câu hỏi: "Điểm danh là gì?"
→ Viết lại: "Điểm danh là gì?" (giữ nguyên vì đã rõ ràng)

**LƯU Ý QUAN TRỌNG:**
- CHỈ xuất câu hỏi đã viết lại, KHÔNG giải thích
- Không thêm thông tin không có trong lịch sử
- Không thay đổi ý định gốc của người dùng"""),
            ("user", """**Lịch sử chat:**
{history}

**Câu hỏi hiện tại:**
{query}

Hãy viết lại câu hỏi (chỉ xuất câu viết lại, không giải thích):""")
        ])
    
    def should_rewrite(self, query: str, history_count: int) -> bool:
        """
        Quyết định có cần viết lại query không
        
        Args:
            query: Câu hỏi của người dùng
            history_count: Số lượng messages trong lịch sử
        
        Returns:
            True nếu nên viết lại
        """
        # Không có lịch sử → không cần rewrite
        if history_count == 0:
            return False
        
        # Danh sách từ khóa chỉ context mơ hồ
        ambiguous_words = [
            # Đại từ
            'họ', 'đó', 'này', 'kia', 'ấy', 'nó', 'nó', 
            # Từ chỉ định mơ hồ
            'thế', 'vậy', 'thì sao', 'như thế nào', 'ra sao',
            # Câu ngắn thiếu ngữ cảnh
            'còn', 'và', 'thì', 'nếu', 'khi'
        ]
        
        query_lower = query.lower()
        
        # Nếu câu hỏi ngắn (< 5 từ) và có history → nên rewrite
        word_count = len(query.split())
        if word_count < 5 and history_count > 0:
            return True
        
        # Kiểm tra có từ mơ hồ không
        for word in ambiguous_words:
            if word in query_lower:
                return True
        
        # Câu hỏi rõ ràng và đủ dài → không cần rewrite
        return False
    
    def rewrite(
        self, 
        query: str, 
        history_messages: list[dict],
        force: bool = False
    ) -> Tuple[str, bool]:
        """
        Viết lại câu hỏi dựa trên lịch sử
        
        Args:
            query: Câu hỏi gốc
            history_messages: Danh sách messages [{'role': 'user'/'assistant', 'content': '...'}]
            force: Ép buộc rewrite dù không cần thiết
        
        Returns:
            (rewritten_query, was_rewritten)
        """
        try:
            # Quyết định có cần rewrite không
            if not force and not self.should_rewrite(query, len(history_messages)):
                logger.info(f"Không cần rewrite: '{query}'")
                return query, False
            
            # Format lịch sử
            if not history_messages:
                history_text = "Không có lịch sử chat."
            else:
                history_parts = []
                # Chỉ lấy 4-6 messages gần nhất để tránh quá dài
                recent_messages = history_messages[-6:]
                for msg in recent_messages:
                    role = "Người dùng" if msg['role'] == 'user' else "AI"
                    content = msg['content'][:200]  # Giới hạn độ dài
                    history_parts.append(f"{role}: {content}")
                history_text = "\n".join(history_parts)
            
            # Gọi LLM để rewrite
            prompt = self.rewrite_prompt.format_messages(
                history=history_text,
                query=query
            )
            
            result = self.llm.invoke(prompt)
            rewritten_query = result.content.strip()
            
            # Validate output
            if not rewritten_query or len(rewritten_query) < 3:
                logger.warning(f"Rewrite thất bại, giữ nguyên: '{query}'")
                return query, False
            
            # Loại bỏ quotes nếu có
            rewritten_query = rewritten_query.strip('"').strip("'")
            
            logger.info(f"Rewritten: '{query}' → '{rewritten_query}'")
            return rewritten_query, True
        
        except Exception as e:
            logger.error(f"Lỗi rewrite query: {e}")
            return query, False  # Fallback về query gốc
    
    def rewrite_with_context(
        self,
        query: str,
        last_n_messages: int = 4
    ) -> Tuple[str, bool]:
        """
        Wrapper cho semantic message history
        Sử dụng khi đã có SemanticMessageHistory object
        """
        # Implement nếu cần
        pass


# ============================================
# USAGE EXAMPLE
# ============================================

def example_usage():
    """Ví dụ sử dụng QueryRewriter"""
    
    rewriter = QueryRewriter()
    
    # Case 1: Câu hỏi mơ hồ có context
    history1 = [
        {"role": "user", "content": "Các đề tài nghiên cứu của giảng viên Nguyễn Văn A là gì?"},
        {"role": "assistant", "content": "Giảng viên Nguyễn Văn A có 3 đề tài: AI, ML, DL"}
    ]
    
    query1 = "Nếu là sinh viên thì sao?"
    rewritten1, changed1 = rewriter.rewrite(query1, history1)
    print(f"Original: {query1}")
    print(f"Rewritten: {rewritten1}")
    print(f"Changed: {changed1}\n")
    
    # Case 2: Câu hỏi rõ ràng
    query2 = "Quy định điểm danh là gì?"
    rewritten2, changed2 = rewriter.rewrite(query2, [])
    print(f"Original: {query2}")
    print(f"Rewritten: {rewritten2}")
    print(f"Changed: {changed2}")


if __name__ == "__main__":
    example_usage()