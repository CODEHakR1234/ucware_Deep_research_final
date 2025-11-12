from app.infra.pdf_loader import PdfLoader
from app.infra.llm_engine import LlmEngine
from app.infra.semantic_grouper import get_semantic_grouper
from .guide_graph_builder import GuideGraphBuilder, GuideState

# 싱글턴 그래프 인스턴스 생성
_graph = GuideGraphBuilder(
    loader=PdfLoader(),
    grouper=get_semantic_grouper(),
    llm=LlmEngine()
).build()

class GuideServiceGraph:
    """튜토리얼 생성 서비스 그래프.
    
    Attributes:
        graph: 컴파일된 LangGraph 파이프라인.
    """
    
    def __init__(self):
        self.graph = _graph

    async def generate(self, file_id: str, pdf_url: str, lang: str):
        """튜토리얼을 생성한다.
        
        Args:
            file_id: 파일 식별자.
            pdf_url: PDF URL.
            lang: 언어 코드.
            
        Returns:
            튜토리얼 생성 결과 딕셔너리.
        """
        result = await self.graph.ainvoke(
            GuideState(
                file_id=file_id, 
                url=pdf_url, 
                lang=lang
            ),
            config={"recursion_limit": 80},
        )
        
        # summary_service_graph.py와 동일한 방식으로 반환
        body = {
            "file_id": file_id,
            "cached": result.get("cached", False),
            "log": result.get("log", []),
        }
        
        if result.get("error"):
            body["error"] = result["error"]
            return body
            
        body["tutorial"] = result.get("tutorial")
        return body

# 싱글턴 인스턴스
_singleton = GuideServiceGraph()

def get_guide_service() -> GuideServiceGraph:
    """GuideServiceGraph 싱글턴을 반환한다."""
    return _singleton

