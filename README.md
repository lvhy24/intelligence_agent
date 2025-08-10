# intelligence_agent
设计思路：
一、智能体结构与基本思路：
  构建一个LocationBehaviorAgent类，其中forward()是智能体核心方法，作为一个任务调度器，根据任务类型调用不同的处理方法。可供forward()调度的处理方法有handle_recommendation()处理地点推荐的方法，handle_review_writing()处理评论生成的方法。
  在handle_recommendation()处理地点推荐的方法中，调用analyze_user_preferences()分析用户偏好，format_location_ratings()获取地点历史评分信息，利用上述方法得到的结果构建可以发送给llm的提示信息。最后通过parse_recommendation_response()方法对llm生成的文本进行解析，得到地点推荐结果。
  在handle_review_writing()处理评论生成的方法中，调用format_user_history()获取用户历史评价，analyze_user_preferences()获取用户偏好，format_location_history()获取地点历史评价，利用上述方法得到的结果构建可以发送给llm的提示信息。最后通过parse_review_response()对llm生成的文本进行解析，得到评论生成结果。
二、一些具体方法细节：
  1.handle_recommendation()：调用相关方法结果构建提示信息给llm，对llm生成的文本进行解析，得到地点推荐结果。如果llm文本生成失败，则返回原始地点列表的顺序作为兜底方案。
  2.handle_review_writing()：调用相关方法结果构建提示信息给llm，对llm生成的文本进行解析，得到评论生成结果。如果llm文本生成失败，则返回"stars": 4, "review": "体验良好，推荐尝试"作为兜底方案。
  3.analyze_user_preferences()：分析用户历史评价，计算用户历史评价平均星级、四星以上评分的比例，统计用户偏好的地点类型以及到访频次，以一定格式生成一段对用户偏好的文本描述供llm参考。
  4.format_user_history()：取用户最近3条历史评价，以一定格式生成一段历史评价集合供llm模仿。
  5.format_location_ratings()：分析地点历史评分，计算地点历史平均评分，生成文本信息供llm参考。
  6.format_location_history()：取地点最近3条历史评价，以一定格式生成一段历史评价集合供llm模仿。
  7.parse_recommendation_response()：如果对llm文本解析失败，则返回原始地点列表的顺序作为兜底方案。
  8.parse_review_response()：llm文本解析失败时，尝试直接在文本中查找评分数字作为地点评分，移除评分后的文本作为地点评论。如果尝试直接在文本中查找评分数字的操作也失败，则取四星作为地点评分，生成的文本直接作为地点评论。
