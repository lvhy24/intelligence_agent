# agent.py
from agentsociety.agent import IndividualAgentBase
from typing import Any
import asyncio#异步编程
import re#正则表达式
import logging#日志记录
import nltk#自然语言处理，包括用于情感分析的词典
try:#尝试查找nltk的vader_lexicon词典，如果没找到，就自动下载
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class LocationBehaviorAgent(IndividualAgentBase):
    """
    城市居民地点行为建模智能体
    处理地点选择推荐和地点评论生成两大任务
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs) #初始化父类
        self.logger = logging.getLogger(__name__)#创建一个日志对象

    async def forward(self, task_context: dict[str, Any]):#一个任务调度器，根据任务类型调用不同的处理方法
        # 提取任务信息
        target = task_context["target"]#接受任务目标：地点推荐or评论生成
        user_id = task_context["user_id"]#提取用户ID
        
        # 获取交互工具
        interaction_tool = self.toolbox.get_tool_object("uir")#获取名称为uir的交互工具
        #根据任务目标调用不同的处理方法
        if target == "recommendation":  
            return await self.handle_recommendation(task_context, interaction_tool)
        elif target == "review_writing":
            return await self.handle_review_writing(task_context, interaction_tool)
        else:
            raise ValueError(f"未知任务类型: {target}")

    async def handle_recommendation(self, task_context: dict, tool: Any):
        """处理地点推荐任务"""
        user_id = task_context["user_id"]
        candidate_list = task_context["candidate_list"]#提取候选地点列表
        #从 task_context 字典中提取 candidate_category 字段的值，表示候选地点的类别。
        #如果 task_context 中没有 candidate_category 字段，则使用默认值 "location"
        candidate_category = task_context.get("candidate_category", "location")
        
        # 获取用户信息
        user_info = tool.get_user(user_id) or {}
        # 获取用户历史评价
        user_reviews = tool.get_reviews(user_id=user_id) or []
        
        # 获取候选地点信息
        candidate_locations = []
        for loc_id in candidate_list:
            loc_info = tool.get_item(loc_id) or {}
            candidate_locations.append({
                "id": loc_id,
                "name": loc_info.get("item_name", "未知地点"),
                "type": loc_info.get("category", candidate_category)
            })
        
        # 分析用户偏好
        user_preferences = self.analyze_user_preferences(user_reviews, tool)

        # 获取地点评分信息（新增部分）
        location_ratings = self.format_location_ratings(candidate_locations, tool)
        
        # 构建提示
        messages = [
            {
                "role": "system", 
                "content": "您是一个城市居民行为建模专家，需要根据居民历史行为预测其对地点的偏好。"
            },
            {
                "role": "user", 
                "content": (
                    f"## 居民信息\n"
                    f"ID: {user_id}\n"
                    f"{user_preferences}\n\n"
                    f"## 候选地点列表\n"
                    f"{self.format_location_list(candidate_locations)}\n\n"
                    f"## 地点历史评分\n"
                    f"{location_ratings}\n\n"
                    f"## 任务要求\n"
                    f"1. 基于居民历史行为和偏好，预测对候选地点的兴趣程度\n"
                    f"2. 考虑地点的历史评分数据，评分高的地点通常更受欢迎\n"
                    f"3. 将候选地点按推荐优先级排序（兴趣最高的排在最前）\n"
                    f"4. 返回格式：推荐排序的地点ID列表，如: [ID1, ID2, ID3,...]"
                )
            }
        ]
        
        # 调用LLM生成推荐
        try:
            response = await self.llm.atext_request(messages)
            return self.parse_recommendation_response(response, candidate_list)
        except Exception as e:
            self.logger.error(f"推荐任务处理失败: {str(e)}")
            return {"item_list": candidate_list}  # 返回原始顺序作为后备

    async def handle_review_writing(self, task_context: dict, tool: Any):
        """处理地点评论生成任务"""
        user_id = task_context["user_id"]
        location_id = task_context["item_id"]
        
        # 获取用户信息
        user_info = tool.get_user(user_id) or {}
        
        # 获取地点信息
        location_info = tool.get_item(location_id) or {}
        
        # 获取用户历史评价
        user_reviews = tool.get_reviews(user_id=user_id) or []
        
        # 获取地点相关评价
        location_reviews = tool.get_reviews(item_id=location_id) or []
                
        # 分析用户偏好
        user_preferences = self.analyze_user_preferences(user_reviews, tool)

        # 构建提示
        messages = [
            {
                "role": "system", 
                "content": "您是一位城市居民，需要为最近访问的地点撰写评价。"
            },
            {
                "role": "user", 
                "content": (
                    f"## 居民信息和用户评价\n"
                    f"ID: {user_id}\n"
                    f"{self.format_user_history(user_reviews)}\n\n"
                    f"## 用户偏好\n"
                    f"{user_preferences}\n\n"
                    f"## 地点信息\n"
                    f"{self.format_location_info(location_info)}\n\n"
                    f"## 地点评价\n"
                    f"{self.format_location_history(location_reviews)}\n\n"
                    f"## 任务要求\n"
                    f"1. 生成1-5星的评分（整数）\n"
                    f"2. 撰写50-100字的评论文本\n"
                    f"3. 评价应依据用户历史评价和地点历史评价，反映真实用户体验\n"
                    f"4. 返回格式：评分: [1-5], 评价: [评论文本]"
                )
            }
        ]
        
        # 调用LLM生成评价
        try:
            response = await self.llm.atext_request(messages)#把构建好的提示发送给LLM
            return self.parse_review_response(response)
        except Exception as e:
            self.logger.error(f"评价生成失败: {str(e)}")
            return {"stars": 4, "review": "体验良好，推荐尝试"}  # 默认返回

    def analyze_user_preferences(self, reviews: list, tool: Any) -> str:
        """分析用户历史评价提取偏好"""
        if not reviews:
            return "该居民暂无历史评价数据"
        
        # 分析评分倾向
        avg_rating = sum(r.get('stars', 3) for r in reviews) / len(reviews)#计算用户历史评价的平均星级
        positive_ratio = sum(1 for r in reviews if r.get('stars', 0) >= 4) / len(reviews)#计算用户四星以上评分的比例
        
        # 分析偏好关键词
        preferences = set()#集合，存放了用户偏好的地点类型
        location_types = {}#字典，存放了用户偏好的地点类型及其出现频次
        
        for review in reviews:
            text = review.get('review', '').lower()
            category_map = {
                "自然与户外": [
                    "公园", "自然", "森林", "湖泊", "河流", "海滩", "山", "步道", "花园", 
                    "绿地", "绿化", "植物园", "露营", "观鸟", "钓鱼", "野餐", "登山", "徒步", "骑行", 
                    "观星", "日出", "日落", "海岸线", "瀑布", "峡谷"
                ],
                
                "文化与历史": [
                    "博物馆", "历史", "遗址", "纪念碑", "寺庙", "教堂", "古街", "古镇", 
                    "美术馆", "画廊", "剧院", "音乐厅", "文化", "故居", "古城墙", "考古", 
                    "传统工艺", "民俗村", "文化街区", "艺术"
                ],
                
                "餐饮美食": [
                    "餐厅", "美食", "咖啡", "小吃", "夜市", "酒吧", "茶馆", "甜品", 
                    "大排档", "火锅", "烧烤", "网红店", "老字号", "米其林", 
                    "特色菜", "美食城", "料理店"
                ],
                
                "购物消费": [
                    "商场", "购物", "百货", "商业街", "免税店", "市场", "集市", "手工艺", 
                    "特产", "古董", "超市", "便利店", "品牌店", "精品店", "商店"
                ],
                
                "休闲娱乐": [
                    "电影院", "KTV", "游乐场", "电玩城", "桌游", "网吧", "水疗", 
                    "按摩", "温泉", "桑拿", "足浴", "美容", "棋牌室", "酒吧街", "主题公园", 
                    "乐园"
                ],
                
                "运动健康": [
                    "体育馆", "健身", "游泳池", "球场", "滑雪场", "高尔夫", "攀岩", 
                    "瑜伽馆", "跑道", "自行车", "滑冰场", "潜水", "冲浪", 
                    "马术", "运动公园"
                ],
                
                "城市设施": [
                    "机场", "车站", "地铁", "公交", "码头", "停车场", "政府", "邮局", 
                    "银行", "派出所", "消防局", "医院", "诊所", "药房", "学校", "幼儿园", 
                    "图书馆", "科技馆", "天文馆", "游客中心", "厕所", "广场"
                ],
                
                "住宿场所": [
                    "酒店", "民宿", "旅馆", "青旅", "度假村", "别墅", "公寓", "客栈", 
                    "露营地", "房车营地", "酒店", "海景房"
                ]
            }
            for category, keywords in category_map.items():
                # 检查关键词
                if any(f'{keyword}' in f'{text}' for keyword in keywords):
                    preferences.add(category)
                    loc_type = category
                    location_types[loc_type] = location_types.get(loc_type, 0) + 1

        # 偏好描述
        pref_text = "、".join(preferences) if preferences else "无明显偏好"
        type_text = ", ".join(
            f"{k}({v})" for k, v in sorted(location_types.items(), key=lambda x: x[1], reverse=True)[:5]#返回出现频次前五名的地点
        ) if location_types else "无类型数据"
        
        return (
            f"历史行为分析：平均评分{avg_rating:.1f}星，{positive_ratio:.0%}评价为积极\n"
            f"偏好类型：{pref_text}\n"
            f"高频访问类型：{type_text}"
        )

    def format_location_list(self, locations: list) -> str:
        """格式化候选地点列表"""
        return "\n".join(
            f"- {loc['name']} ({loc['type']}, ID: {loc['id']})"
            for loc in locations
        )

    def get_location_avg_rating(self, location_id: str, tool: Any) -> float:
        """获取指定地点的历史评价平均分"""
        # 获取该地点的所有评价
        reviews = tool.get_reviews(item_id=location_id) or []
        if not reviews:
            return 0.0
        # 计算平均分
        total_rating = sum(review.get('stars', 0) for review in reviews)
        return round(total_rating / len(reviews), 1)

    def format_location_ratings(self, locations: list, tool: Any) -> str:
        """格式化地点评分信息"""
        rating_info = []
        for loc in locations:
            avg_rating = self.get_location_avg_rating(loc['id'], tool)
            rating_text = f"{avg_rating}星" if avg_rating > 0 else "暂无评分"
            rating_info.append(f"- {loc['name']}: {rating_text}")
        
        return "\n".join(rating_info)

    def format_user_history(self, reviews: list) -> str:
        """格式化用户历史评价（供LLM模仿）"""
        if not reviews:
            return "该居民暂无历史评价"
        
        # 取最近3条
        recent_reviews = sorted(reviews, key=lambda x: x.get('timestamp', 0), reverse=True)[:3]
        
        sample_reviews = "\n".join(
            f"  - {r.get('item_id', '未知地点')}: {r.get('stars', '?')}星, {r.get('review', '')[:50]}..."
            for r in recent_reviews
        )
        return f"近期评价：\n{sample_reviews}"

    def format_location_history(self, reviews: list) -> str:
        """格式化地点历史评价"""
        if not reviews:
            return "该地点暂无历史评价"
        
        # 取最近5条
        recent_reviews = sorted(reviews, key=lambda x: x.get('timestamp', 0), reverse=True)[:5]
        
        sample_reviews = "\n".join(
            f"  - {r.get('item_id', '未知地点')}: {r.get('stars', '?')}星, {r.get('review', '')[:100]}..."
            for r in recent_reviews
        )
        return f"近期评价：\n{sample_reviews}"

    def format_location_info(self, location: dict) -> str:
        """格式化地点信息"""
        if not location:
            return "地点信息缺失"
        
        info = [
            f"名称: {location.get('item_name', '未知地点')}",
            f"类型: {location.get('category', '未知类型')}",
            f"描述: {location.get('description', '无描述信息')[:100]}..."
        ]
        return "\n".join(info)

    def parse_recommendation_response(self, response: str, candidates: list) -> dict:
        """解析推荐任务响应"""
        try:
            # JSON格式
            if response.startswith("[") and response.endswith("]"):#如果response是json格式的列表
                import json
                found_ids = json.loads(response)#json.loads()函数将JSON字符串解析为Python对象
                if isinstance(found_ids, list):
                    return self.validate_recommendation_list(found_ids, candidates)
            
            # 逗号分隔
            if "," in response:#如果response是逗号分隔的字符串
                found_ids = [id.strip() for id in response.split(",")]
                return self.validate_recommendation_list(found_ids, candidates)
            
            # 正则表达式
            id_pattern = r"[A-Za-z0-9_-]{4,}"#如果response是自由文本格式的字符串，使用正则表达式提取ID，匹配至少4位的字母、数字、下划线或短横线
            found_ids = re.findall(id_pattern, response) # 提取所有匹配的 ID，返回匹配到的ID列表
            return self.validate_recommendation_list(found_ids, candidates)
        
        except Exception as e:
            self.logger.warning(f"推荐解析失败: {str(e)}")
            return {"item_list": candidates}

    def validate_recommendation_list(self, found_ids: list, candidates: list) -> dict:
        """验证并补全推荐列表"""
        valid_ids = []
        # 添加找到的有效ID
        for id in found_ids:
            if id in candidates and id not in valid_ids:
                valid_ids.append(id)
        # 添加缺失的候选ID
        for candidate in candidates:
            if candidate not in valid_ids:
                valid_ids.append(candidate)
        return {"item_list": valid_ids}

    def parse_review_response(self, response: str) -> dict:
        """准确解析评论任务响应，提取评分和评论文本"""
        # 产生文本的标准格式 "评分: X, 评价: Y"
        pattern = r"评分\s*[:：]\s*(\d)[\s,;，；]*评价\s*[:：]\s*(.+)"//#按照评分: X, 评价: Y的格式提取评分和评论
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                stars = int(match.group(1).strip())#提取评分
                review = match.group(2).strip()#提取评论文本
                # 验证评分范围
                if 1 <= stars <= 5:
                    return {"stars": stars, "review": review}
            except (ValueError, IndexError):
                pass
        
        # 模式不匹配后的处理：尝试直接在文本中查找评分数字
        digit_match = re.search(r"\b([1-5])\b", response)
        if digit_match:
            try:
                stars = int(digit_match.group(1))
                # 移除评分数字后的文本作为评论
                review = re.sub(r"\b[1-5]\b", "", response, count=1).strip()
                return {"stars": stars, "review": review}
            except (ValueError, IndexError):
                pass
        
        # 最终备选方案
        return {"stars": 4, "review": response}
      
