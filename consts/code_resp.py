from vo.customer_err import CustomError

Err_System = CustomError(10000, "系统异常")
Err_Embedder_Info = CustomError(10001, "向量化模型不存在")