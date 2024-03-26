Task: For Ideal Customer Profile Research (ICP framework), I was asked to show what “properties” or “features” could increase conversion chances and which could decrease them.

Features examples:
The company is from “New York”
The company website uses the “New Relic” tool
Similarweb categorized the company industry as “Finance”
NAIC description for a company contains the word: “Manufacturing”
GPT categorizes the contact company business model as: “non-profit”
Company contact visited page “blog/article_name”
 

Challenges:
There are fewer leads (just about 40k) in funnel input, but we have 20k different features. Standard (usually Python libraries) approaches couldn’t handle such many dimensio (every feature is a dimension)
Every day, new inputs exist. The model should be updated
Sales and Marketing teams alike to participate in adding new features and feature mapping.

Solution:
After taking all the input, we focused on data engineering. We (one intern data engineer and I) split all transformation into three steps
OLAP with Feature mappings. Models where we combine companies and features (20k dimensional OLAP cube)
Sales funnel Flat model: Every company has a column with the date of the main event: email submission, discovery call, demo date, proposal date, and won date.
Mathematical model. Naive Bayes classifier that checks if any feature 

Audit.
Since the model is kinda complex, it were reviewed by an independent analyst. I wrote this little math paper that explains the statistical model behind it:
