intermediate_query_prefix = """Given a product description and a review, answer whether the product would be suitable for kids. Answer with ONLY "Yes" or "No". Here are some examples first:
Example:
description: "1900 Classical Violin Music"
reviewText: "This is a comforting track that gives me peaceful time away from my children"
Answer: "No".

Example:
description: "Smooth funky jazz music!"
reviewText: "My son really loves listening to this!"
Answer: "Yes"

Here is the actual review:"""

basic_query_prefix = """Given the following ReviewText, answer whether the sentiment associated is "Positive" or "Negative". Answer with ONLY "Positive", "Negative", or "Neutral". Here are some examples first:
Example: 
ReviewText: "Very boring watch, the performances in this movie were horrible"
Answer: "Negative".

Example: 
ReviewText: "Entertaining fun time with family!"
Answer: "Positive"

Example: 
ReviewText: "Not sure how I feel about this movie..."
Answer: "Neutral"

Here is the actual review:"""
