# Naive Bayes Classifier for Companies Classification.

## Input data
1. Table ```dim_companies__with_all_factors``` with companies and factors stored as an array. Each Factor has a datasource prefix representing the source of data. 
2. Table ```dim_customer__funnel_metrics_flat``` with funnel data for the companies.

## Brief Algorithm Explanation
In the most simple case the algorithm consists of the following steps:
#### Preprocessing
1. Determine the working sample.<br>
The current implementation uses sample of companies from ```dim_customer__funnel_metrics_flat``` with ```first_discovery_call_book_completed_datetime``` not equals to ```null```.
2. Determine target variable and it's possible values (classes).<br>
The current implementation uses binary target variable $c$ that equals to $0$ if ```won_date``` is ```null``` and $1$ otherwise.
3. Determine factors that are included in the analysis.<br>
The current implementation uses significant features that satisfy the limitations of Pearson's chi-squared test ($\chi^2$). Among the “rare” factors with any of expected frequencies is less than 3, significant factors are selected according to a separate custom rule.
4. For each company create an array of binary features, denoted as $x_i \in \{0,1\}$, where $1$ represents the presence of a specific factor in the array, and $0$ represents its absence. The modified algorithm identifies the data source prefixes with maximum count of linked tags for each company and treats these prefixes as categorical features themselves, with set of tags considered as a set of possible categories.
#### Model fitting 
5. Find the frequencies of all indicated features among the samples assigned to each class $c$.
6. Estimate conditional probabilities of features given each class $c$ using smoothing parameter $\alpha$ (for example, $1$).
7. Estimate the probabilities of classes $c$.
#### Target variable prediction 
8. Given array of factors calculate for each class $c$ it's weight using Bayes’ theorem.
9. Find the _argmax_ over $c$ to define the most probable class.
10. Normalize weights to estimate probabilities of classes given set of features. 

## Naive Bayes Classifier for Categorical Features.

### Assumptions
The fundamental Naive Bayes assumption is that each feature makes an __independent__ and __equal__ contribution to the target variable.

In-fact, the independence assumption is never correct but often works well in practice.
The better the data aligns with the assumptions, the higher the quality of the classification.

### Theory
[Theory reference](https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes)

Bayes’ theorem states the following relationship, given class variable $y$ and dependent feature vector $x_1$ through $x_n$:
$$P(y | x_1, x_2, \ldots, x_n) = \frac{P(y)P(x_1, x_2, \ldots, x_n | y)}{P(x_1, x_2, \ldots, x_n)}.$$
>_Example_:
>$$P(\text{ if won } | \text{ if enterprise } \text{and} \text{ if using trade desk }) =$$
>$$ = \frac{P(\text{ if won })P(\text{ if enterprise } \text{and} \text{ if using trade desk } | \text{ if won})}{P(\text{ if enterprise } \text{and} \text{ if using trade desk })}$$

Using the naive <b>conditional independence assumption</b> that
$$P(x_1, x_2, \ldots, x_n | y) = P(x_1 | y)P(x_2 | y)\ldots P(x_n | y),$$
relationship is simplified to
$$P(y | x_1, x_2, \ldots, x_n) = \frac{P(x_1 | y)P(x_2 | y)\ldots P(x_n | y) P(y)}{P(x_1,x_2,\ldots,x_n)}.$$
>_Example_:
>$$P(\text{ if won } | \text{ if enterprise } \text{and} \text{ if using trade desk }) =$$
>$$ = \frac{P(\text{ if won })P(\text{ if enterprise }| \text{ if won }) P(\text{ if using trade desk } | \text{ if won })}{P(\text{ if enterprise } \text{and} \text{ if using trade desk })},$$
>$$P(\text{ if lost } | \text{ if enterprise } \text{and} \text{ if using trade desk }) =$$
>$$ = \frac{P(\text{ if lost })P(\text{ if enterprise }| \text{ if lost }) P(\text{ if using trade desk } | \text{ if lost })}{P(\text{ if enterprise } \text{and} \text{ if using trade desk })}.$$

Since $P(x_1,\ldots,x_n)$ is constant given the input, we can use the following classification rule:
$$\hat y = \argmax_y P(y)\prod_{i=1}^n P(x_i | y),$$
or 
$$\hat y = \argmax_y\left(\ln P(y) + \sum_{i=1}^n \ln P(x_i | y)\right).$$
>_Example_:
>$$\hat y = \argmax_{\text{ result = won, lost}} = P(\text{ result })P(\text{ if enterprise }| \text{ result }) P(\text{ if using trade desk } | \text{ result }).$$

### Model fitting

The probability of category $t$ in feature $i$ given class $c$ is The estimated as:
$$\hat P(x_i = t | y = c) = \frac{N_{tic}}{N_c},$$
where $N_{tic} = |\{j \in J | x_{ij} = t, y_j = c\}|$ is the number of times category $t$ appears in the samples $x_i$, which belong to class $c$, $N_{c} = |\{j \in J | y_j = c\}|$ is the number of samples with class $c$.
>_Example_:
>$$\hat P(\text{ if using trade desk } | \text{ if won }) = \frac{\text{count}(\text{ if using trade desk } \text{and} \text{ if won })}{\text{count}(\text{ if won })}.$$


Additionally model performance could be improved by adding a smoothing parameter $\alpha$:
$$\tilde P(x_i = t | y = c; \alpha) = \frac{N_{tic} + \alpha}{N_c + \alpha n_{i}},$$
where $n_{i}$ is the number of available categories of feature $i$.
>_Example_:
>$$\hat P(\text{ if using trade desk } | \text{ if won }) = \frac{\text{count}(\text{ if using trade desk }  \text{and} \text{ if won }) + 2}{\text{count}(\text{ if won }) + 4}.$$

>_Note_:
>The smoothing parameter $\alpha$, also known as Laplace smoothing or add-one smoothing, is used in categorical Naive Bayes classification to address the issue of zero probabilities. In categorical Naive Bayes, when a feature-label combination is unseen in the training data, the probability estimate becomes zero, which can cause problems in further likelihood estimation.
>
>$\alpha$ is added to the observed counts of each category to ensure that no probability estimate is zero. This is important for two reasons:
>
>1. Avoiding Zero Probabilities:
>Without smoothing (alpha = 0), if a particular category of a feature is not present in the training data for a given class, the probability of that class given that feature becomes zero. Smoothing prevents this by assigning a small non-zero probability to unseen categories.
>
>The specific choice of $\alpha$ can impact the balance between smoothing and reliance on observed data.

>_Note_:
> For the special case of $n_i = 2$ (binary $x$), $\tilde P(x=t | c; \alpha=1)$ is a clostd form Bayes estimator for $p$, when using the Uniform distribution as a conjugate prior distribution (or Beta distribution $\Beta(1,1) = U(0,1)$). Uniform distribution assigns equal probability to all values of $p$ within the range $[0, 1]$. [ref.](https://en.wikipedia.org/wiki/Beta_distribution#Bayesian_inference)

>_Note_:
>Aslo, for the special case of $n_i = 2$ (binary $x$), $\tilde P(x=t | c; \alpha=z^2/2)$ is the centre-point of the approximate binomial confidence interval, where $z = \Phi^{-1}\left(1 - \alpha^*/2 \right)$ is the quantile of a standard normal distribution. For example, a $95\%$ confidence interval requires $\alpha^* = 0.05$, thereby producing $z = 1.96$. Taking $z = 2$ produces the "add 2 successes and 2 failures" interval with the confidence $\sim 95.45\%$. [ref.](https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Agresti%E2%80%93Coull_interval)

### Estimate target class probabilities

Given $\hat P(x_i = t | y = c)$ and $\hat P(y)$ calculated,  predictions can be made as
$$\hat y = \text{argmax}_y\left(\ln \hat P(y) + \sum_{i=1}^n \ln \hat P(x_i | y)\right).$$

When using Bayes’ theorem, the computed values for $\hat P(y | x)$ may not accurately represent probabilities due to dependencies among features. However, they can still be used as class weights:

$$W(y | x_1, x_2, \ldots, x_n) = \frac{\hat P(y) \hat P(x_1 | y)\hat P(x_2 | y)\ldots \hat P(x_n | y)}{\hat P(x_1,x_2\ldots x_n)}.$$

For the estimation of "shared" probabilities $P(x_1, \ldots, x_n)$, $y_j=0,1$, an additional independence assumption can be considered 
$$\hat P(x_1,x_2,\ldots,x_n) = \hat P(x_1)\hat P(x_2)\ldots\hat P(x_n)$$

Subsequently, by normalizing these values, one can derive approximate probability estimates:
$$\hat P(y_i | x) \approx W(y_i | x) \Biggm/ \sum_{j}W(y_j | x).$$

That leads to the same results as normalizing numerators only:

$$\hat P(y_i | x) \approx w(y_i | x) \Biggm/ \sum_{j}w(y_j | x),$$
where $w(y_i | x) = \hat P(y) \hat P(x_1 | y)\hat P(x_2 | y)\ldots \hat P(x_n | y).$

For the calculation stability usually logarithmic values are used instead
$$\ln \left(w(y | x_1, x_2, \ldots, x_n)\right) =
\ln \left(\hat P(y)\right)
+ \sum_{i=1}^{n} \ln \left(\hat P(x_i | y)\right).$$

### Example 

In the sample of companies with completed first discovery calls, which includes 2635 companies, 402 of them are attributed as 'is_enterprise.' Out of these 402 companies, 20 of them have a 'won_date.' Among the 2233 companies attributed as small or mid-market, there are 33 companies that have a 'won_date.' These frequencies can be visualized in a 2x2 contingency table:
|               | if_won  | if_lost | Total |
| ---: | :---: | :---: | :---: |
| if_enterprise | 20  | 382  | 382  |
| if_not_enterprise | 33  | 2200  | 2233  |
| __Total__ | 53   | 2582 | 2635 |

For the binary 'if_won' target variable, we will denote $c=1$ for 'won' and $c=0$ for 'lost' class. The estimated conditional probabilities for $c$ are as follows:

$$P(c=1) = \frac{53}{2635} \approx .02,\quad P(c=0) \approx .98.$$

The estimated conditional probabilities without smoothing are:
$$\hat  P(t=\text{if enterprise} | 1 ) = \frac{20}{53} \approx .38,\quad \hat  P(t=\text{if enterprise} | 0 ) = \frac{382}{2582} \approx .15.$$

These frequencies of the second tag can be visualized in a 2x2 contingency table:
|      | if_won  | if_lost | Total |
| ---: | :---: | :---: | :---: |
| if_from_new_york | 8  | 149  | 157  |
| if_not_from_new_york | 45  | 2433  | 2478  |
| __Total__ | 53   | 2582 | 2635 |

The estimated conditional probabilities without smoothing are:
$$\hat  P(t=\text{if new york} | 1 ) = \frac{8}{53} \approx .15,\quad \hat  P(t=\text{if new york} | 0 ) = \frac{149}{2582} \approx .06.$$

Calculating the weights:
$$w(c=1 | t=\text{true}) = \hat P(c=1) \hat P(t_1=\text{true} | 1 ) \hat P(t_2=\text{true} | 1 ) = 0.02 \cdot 0.38 \cdot 0.15 \approx 1.15\text{E-3},$$
$$w(c=0 | t=\text{true}) = \hat P(c=0) \hat P(t_1=\text{true} | 0 ) \hat P(t_2=\text{true} | 0 ) = 0.98 \cdot 0.15 \cdot 0.06 \approx 8.37\text{E-3}.$$

Normalizing these weights results in "probabilities" as: 
$$ \hat P(c=1 | t=\text{true}) \approx 12\%.$$

The estimated probability of winning if the company is an enterprise and from New York is approximately 12%.

## Performance metrics
For choosing appropriate smoothing parameter $\alpha$ there are number of ways to compare the prediction quality for categorical classifier. These methods include precision, recall, F1-score, and $F_2$-score. The $F_2$-score, where $\beta = 2$, assigns higher weight to recall compared to precision, making it the preferred metric in this context [ref.](https://en.wikipedia.org/wiki/Precision_and_recall) 
 1. Precision $$precision = \frac{relevant  retrieved  instances}{all  retrieved  instances}$$
 2. Recall $$recall = \frac{relevant  retrieved  instances}{all  relevant  instances}$$
 3. F1-score $$F = 2 \cdot \frac {precision \cdot recall} {precision + recall}$$
 4. $F_\beta$-score $$F_\beta = (1 + \beta^2) \cdot \frac {precision \cdot recall} {\beta^2 \cdot precision + recall}$$
 For example, $F_\beta$, where $\beta = 2$, weights recall higher than precision.
 
 And many others...

## Feature dependency issues

In this section, we discuss feature dependency issues and their impact on classification.

### The problem with dependent features:

There are two different tags in the ```dim_companies__with_all_factors``` dataset, there are two distinct tags that provide the same information. They are ```event_campaign:brand_search``` and ```session_campaign:brand_search```. Both have the same contingency table:

| category $t$ | Success  | Failure | Total |
| ---:  | :---: | :---: | :---: |
| has this tag     | 5      | 12   |   17 |
| does not have this tag | 107    | 3396 | 3503 |
| Total | 112   | 3408 | 3520 |

The estimated class probabilities are:
$$P(c=1) = \frac{112}{3520} \approx .032,\quad P(c=0) \approx .968.$$

The estimated category probabilities are:
$$\hat P(t=\text{true}) = \frac{17}{3520} \approx .005,\quad \hat P(t=\text{false}) = \frac{3503}{3520} \approx .995.$$

The estimated conditional feature probabilities given $c = 1$ (_Success_) are
$$\hat  P(t=\text{true} | 1 ) = \frac{5}{112} \approx .045,\quad \hat  P(t=\text{false} | 1 ) = \frac{107}{112} \approx .955.$$

Thus, according to the Bayes’ theorem, probabilities of $c = 1$ given each category $t$ could be The estimated as:
$$\hat P(c=1 | t=\text{true}) = \frac{\hat P(c=1) \hat P(t=\text{true} | c= 1 )}{\hat P(t=\text{true})} \approx 29\%,\quad \hat P(c=0 | t=\text{true}) \approx 71\%.$$

This is the right answer.

Going forward with algorithm 
$$w(c=1 | t=\text{true}) = \hat P(c=1) \hat P(t_1=\text{true} | 1 ) \hat P(t_2=\text{true} | 1 ) = 0.032 \cdot 0.045^2 \approx 6.34\text{E-5},$$
$$w(c=0 | t=\text{true}) = \hat P(c=0) \hat P(t_1=\text{true} | 0 ) \hat P(t_2=\text{true} | 0 ) = 0.968 \cdot 0.004^2 \approx 1.20\text{E-5}.$$

Normalizing these weights obtain "probabilities" as 
$$ \hat P(c=1 | t=\text{true}) \approx 84\% ,\quad \hat P(c=0 | t=\text{true}) \approx 16\%.$$

The predicted class $c$ has been changed! Of course the second result is wrong as we could not use conditional independence assumption. However, both features influence the target variable with the same degree of reliability.

If there were a third "copy" of this factor in the array, the calculated probability of winning given only this multiplied factor would increase to 99\%.

### Categorical feature _vs_ set of binary features (Mutually exclusive factors)

Another example of fully dependent factors is the case of mutually exclusive factors. For instance, if a company is categorized as 'small,' it logically follows that it cannot simultaneously be classified as 'big' or 'medium-sized.

For such tags the simplest case, in which we consider only binary features $x_i \in {1,0}$, leads to biases in the model's predictions.

Consider the following distribution: we have the datasource prefix ```page_s_viewed_count``` and possible categories:
```from_1_to_5```, ```from_6_to_9``` and ```from_10```

| category $t$ | Class c=1  | Class c=0 | Total |
| ---: | :---: | :---: | :---: |
| from_1_to_5 | 40  | 2455 | 2495 |
| from_6_to_9 | 42  | 802  | 844 |
| from_10 | 30  | 179  | 209 |
| Total | 112  | 3436  | 3548 |

The estimated class probabilities are:
$$P(c=1) = \frac{112}{3548} \approx .032,\quad P(c=0) \approx .968.$$

The estimated category probabilities are:
$$\hat P(t=\text{from 1 to 5}) = \frac{2495}{3548} \approx .70,\quad \hat P(t=\text{from 6 to 9}) = \frac{844}{3548} \approx .24,\quad \hat  P(t=\text{from 10}) = \frac{209}{3548} \approx .06.$$

The estimated conditional category probabilities given $c = 1$ (_Success_) are

$$\hat P(t=\text{from 1 to 5} | 1 ) = \frac{40}{112} \approx .36,\quad \hat  P(t=\text{from 6 to 9} | 1 ) = \frac{42}{112} \approx .38,\quad \hat  P(t=\text{from 10} | 1 ) = \frac{30}{112} \approx .27.$$

Thus, according to the Bayes’ theorem, probabilities of $c = 1$ given each category $t$ could be The estimated as:
$$\hat P(c=1 | t=\text{from 1 to 5}) = \frac{\hat P(t=\text{from 1 to 5} | c= 1 )\hat P(c=1)}{\hat P(t=\text{from 1 to 5})} \approx .016,$$
$$\hat P( c=1 | t=\text{from 6 to 9}) = \frac{\hat P(t=\text{from 6 to 9} | c= 1 )\hat P(c=1)}{\hat P(t=\text{from 6 to 9})} \approx .050,$$
$$\hat P( c=1 | t=\text{from 10}) = \frac{\hat P(t=\text{from 10} | c= 1 )\hat P(c=1)}{\hat P(t=\text{from 10})} \approx .144.$$

Let's now consider the case when this categorical variable is transformed into three mutually exclusive (and therefore dependent) binary features.

The 2x2 contingency table for each feature and corresponding complement probabilities are:

1. ```from_1_to_5```

| category $t$ | Class c=1  | Class c=0 | Total |
| ---: | :---: | :---: | :---: |
| from_1_to_5 | 40  | 2455 | 2495 |
| $\neg$ from_1_to_5 | 72  | 981  | 1053 |
| Total | 112  | 3436  | 3548 |

$$\hat P(t=\neg from 1 to 5 | c= 1) = \frac{72}{112} \approx .64,\quad \hat P(t=\neg from 1 to 5 | c= 0) = \frac{981}{3436} \approx .29.$$

2. ```from_6_to_9```

| category $t$ | Class c=1  | Class c=0 | Total |
| ---: | :---: | :---: | :---: |
| from_6_to_9 | 42  | 802  | 844 |
| $\neg$ from_6_to_9 | 70 | 2634 | 2704 |
| Total | 112  | 3436  | 3548 |

$$\hat P(t=\neg from 6 to 9| c= 1) = \frac{70}{112} \approx .63,\quad
\hat P(t=\neg from 6 to 9| c= 0) = \frac{2634}{3436} \approx .77.$$

3. ```from_10```

| category $t$ | Class c=1  | Class c=0 | Total |
| ---: | :---: | :---: | :---: |
| from_10 | 30  | 179  | 209 |
| $\neg$ from_10 | 82 | 3257 | 3339 |
| Total | 112  | 3436  | 3548 |

$$\hat P(t=\neg from 10 | c = 1) = \frac{82}{112} \approx .73,\quad
\hat P(t=\neg from 10| c= 0) = \frac{3257}{3436} \approx .95.$$

So the alternative sets of features are $(1, 0, 0)$, $(0, 1, 0)$ and $(0, 0, 1)$
$$w(c= 1 | 1,0,0) = \hat P(c=1)\hat P(t=\text{from 1 to 5} |1 )\hat P(t=\neg \text{from 6 to 9} | 1 )\hat P(t=\neg \text{from 10} | 1 ) \approx 5.16\text{E-3}$$
and 
$$w(c= 0 | 1,0,0) = \hat P(c=0)\hat P(t=\text{from 1 to 5} |0 )\hat P(t=\neg \text{from 6 to 9} | 0 )\hat P(t=\neg \text{from 10} | 0 ) \approx 5.03\text{E-1}.$$

Similarly, obtain the estimates
$$w(c= 1 | 0,1,0) \approx 5.57\text{E-3},$$
$$w(c= 0 | 0,1,0) \approx 6.12\text{E-2},$$
and
$$w(c= 1 | 0,0,1) \approx 3.40\text{E-3},$$
$$w(c= 0 | 0,0,1) \approx 1.10\text{E-2}.$$

Normalizing these weights obtain the following probabilities:
$$\hat P(c= 1 | 1,0,0) = .010,\quad \hat P(c= 1 | 0,1,0) = .083,\quad \hat P(c= 1 | 0,0,1) = .235.$$

The table below illustrates the distinction between categorical and binary input, showing the corresponding probabilities of '_Success_' for each tag:
| factor repr. |  from_1_to_5 | from_6_to_9 | from_10 |
| ---: | :---: | :---: | :---: |
| categorical | 1.6%  | 5.0%  | 14.4% |
| binary tuple | 1.0% | 8.3% | 23.5% |

In both cases, the maximum likelihood estimation of the target is '_Failure_'. Nevertheless, even for a variable with only three possible categories, there is a notable difference in the estimates of the probabilities among these treatments.

<!-- ## Data sources interaction
This section is not finished yet. -->

## Conclusion
In order to improve the Naive Bayes classifier, the main goal is to reduce the dependency between factors.

In the case where individual tags are a priori mutually exclusive, the data source prefix can be thought of as a feature and the individual tags as categories for that feature.

If there are tags with the same or similar information, they should be excluded manually or some suitable method based on expert knowledge on that particular topic should be used.
