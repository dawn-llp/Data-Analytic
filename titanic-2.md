titanic-2
================
ShayleeLi (Liping)
Dec 18, 2016

### -Import and Overview

``` r
library(data.table)
train = fread("train-titanic.csv", sep = ",", header = TRUE, stringsAsFactors = TRUE)
setkey(train)
summary(train)
```

    ##   PassengerId       Survived          Pclass     
    ##  Min.   :  1.0   Min.   :0.0000   Min.   :1.000  
    ##  1st Qu.:223.5   1st Qu.:0.0000   1st Qu.:2.000  
    ##  Median :446.0   Median :0.0000   Median :3.000  
    ##  Mean   :446.0   Mean   :0.3838   Mean   :2.309  
    ##  3rd Qu.:668.5   3rd Qu.:1.0000   3rd Qu.:3.000  
    ##  Max.   :891.0   Max.   :1.0000   Max.   :3.000  
    ##                                                  
    ##                                     Name         Sex           Age       
    ##  Abbing, Mr. Anthony                  :  1   female:314   Min.   : 0.42  
    ##  Abbott, Mr. Rossmore Edward          :  1   male  :577   1st Qu.:20.12  
    ##  Abbott, Mrs. Stanton (Rosa Hunt)     :  1                Median :28.00  
    ##  Abelson, Mr. Samuel                  :  1                Mean   :29.70  
    ##  Abelson, Mrs. Samuel (Hannah Wizosky):  1                3rd Qu.:38.00  
    ##  Adahl, Mr. Mauritz Nils Martin       :  1                Max.   :80.00  
    ##  (Other)                              :885                NA's   :177    
    ##      SibSp           Parch             Ticket         Fare       
    ##  Min.   :0.000   Min.   :0.0000   1601    :  7   Min.   :  0.00  
    ##  1st Qu.:0.000   1st Qu.:0.0000   347082  :  7   1st Qu.:  7.91  
    ##  Median :0.000   Median :0.0000   CA. 2343:  7   Median : 14.45  
    ##  Mean   :0.523   Mean   :0.3816   3101295 :  6   Mean   : 32.20  
    ##  3rd Qu.:1.000   3rd Qu.:0.0000   347088  :  6   3rd Qu.: 31.00  
    ##  Max.   :8.000   Max.   :6.0000   CA 2144 :  6   Max.   :512.33  
    ##                                   (Other) :852                   
    ##          Cabin     Embarked
    ##             :687    :  2   
    ##  B96 B98    :  4   C:168   
    ##  C23 C25 C27:  4   Q: 77   
    ##  G6         :  4   S:644   
    ##  C22 C26    :  3           
    ##  D          :  3           
    ##  (Other)    :186

### -Deal with Age's NAs

Age has 177 missing values. Cabin has 687 NAs and Embarked has 2. We'll can put Cabin and Embarked aside. Age is an important demographic variable, so we can try to fill with predictions.

``` r
train$Embarked = as.character(train$Embarked)  #change data type that can add NA
train[Embarked == "", `:=`(Embarked, NA)]  #set '' in Embarked as missing value
```
``` r
# Tried regression model, R-square is not ideal. So take the tree method.
library(rpart)
library(ggplot2)
library(gridExtra)
age_fit2 = rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked, data = train[!is.na(Age), 
    ], na.action = na.omit, method = "anova", cp = 0.01)
page1 = qplot(train$Age, geom = "histogram", fill = I("turquoise3"), xlab = "Age-raw", 
    ylab = "Numbers of People")
train[is.na(Age), `:=`(Age, predict(age_fit2, newdata = train[is.na(Age)]))]
```
``` r
page2 = qplot(train$Age, geom = "histogram", fill = I("turquoise3"), xlab = "Age-adjusted", 
    ylab = "Numbers of People")
grid.arrange(page1, page2, ncol = 2)
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-3-1.png)

The two distribution of age seem alike. That's what we want.

### -List single variable distribution except Name, Ticket and Cabin

Now, most of variables are factors and numbers. It's like that you finally pick all the vegetables and meat for dinner. Cannot wait to have a raw view.

``` r
psur = qplot(train$Survived, geom = "bar", ylab = "Numbers of People", fill = I("turquoise3"), 
    xlab = "0=Not survived   1=Survived")
ppclass = qplot(train$Pclass, geom = "bar", fill = I("turquoise3"), xlab = "Social-Economic Status", 
    ylab = "Numbers of People")
psex = qplot(train$Sex, geom = "bar", fill = I("turquoise3"), xlab = "Gender", 
    ylab = "Numbers of People")
pss = qplot(train$SibSp, geom = "bar", xlab = "People with x Siblings and Spouse", 
    ylab = "Numbers of People", fill = I("turquoise3"))
ppc = qplot(train$Parch, geom = "bar", xlab = "People with x Parents and Children", 
    ylab = "Numbers of People", fill = I("turquoise3"))
pf = qplot(train$Fare, geom = "histogram", fill = I("turquoise3"), xlab = "Price of Ticket", 
    ylab = "Numbers of People")
pe = qplot(train$Embarked, xlab = "Embarked Port", fill = I("turquoise3"), ylab = "Numbers of People", 
    geom = "bar")
grid.arrange(psur, ppclass, psex, page2, pss, ppc, pf, pe, ncol = 3)
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-4-1.png)

Most people are in their golden ages(20~40). Most people are men. Most people are travel alone. Most people from 3rd class. Most people from Southampton. And most of them lost their lives in that adventure.

### -Interesting Thing about Name: New Name &New Life

Name has many interesting title. Let's grasp these info.

``` r
train$Title = gsub("(.*,)|(\\..*)", "", train$Name)
train$Title = as.factor(train$Title)
train$Survived = as.character(train$Survived)
ggplot(train, aes(x = Title, fill = Survived)) + geom_bar() + coord_flip() + 
    scale_fill_brewer(palette = "BuGn") + ylab("")
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-5-1.png)

The survived "Miss"" is **no** more than married women "Mrs"" in terms of ration. I've been heard that people in Titanic give survivor opportunities to unmarried women. But according to this chart, it seems not ture. We'll dig into it later.

Another phenomenon attracts me is that many people have two names, like "Cumings, Mrs. John Bradley (Florence Briggs Thayer)". Did they change name before this tragedy or after? We don't know. Just have a look.

``` r
train$NewName = grepl("(", train$Name, fixed = TRUE)
train$Survived = as.character(train$Survived)
ggplot(train, aes(x = NewName, fill = Survived)) + geom_bar() + scale_fill_brewer(palette = "BuGn") + 
    ylab("")
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-6-1.png)

The ration looks high for people survived to have a second name.

### -A Quick Look of Correlation and Importance: Women, Young, High Class

``` r
library(GGally)  #Only shows numeric variables
train[Sex == "male", `:=`(nSex, 1)]
train[Sex == "female", `:=`(nSex, 2)]
```

``` r
train[Embarked == "C", `:=`(Port, 1)]
train[Embarked == "Q", `:=`(Port, 2)]
train[Embarked == "S", `:=`(Port, 3)]
```

``` r
train$Survived = as.numeric(train$Survived)
ggcorr(train, label = TRUE, label_alpha = TRUE, name = "Correlation", low = "turquoise3", 
    mid = "white", high = "orangered3")
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-7-1.png)

``` r
library(randomForest)
rf = randomForest(Survived ~ Pclass + Age + nSex + Parch + SibSp + Fare + Port, 
    data = train, na.action = na.omit, ntree = 101, importance = TRUE)
imp = importance(rf, type = 1)
featureImportance = data.frame(Feature = row.names(imp), Importance = imp[, 
    1])
ggplot(featureImportance, aes(x = reorder(Feature, Importance), y = Importance)) + 
    geom_bar(stat = "identity", fill = I("turquoise3")) + coord_flip() + xlab("") + 
    ylab("Importance on Survive")
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-7-2.png)

Women, people of young age and people of higher class are more likely to survive.

### -X with Survived

Take a close look of every independent variable influence on survive. This time add SibSp and Parch as family size

``` r
train$Survived = as.character(train$Survived)
SurPc = ggplot(train, aes(x = Pclass, fill = Survived)) + geom_bar() + scale_fill_brewer(palette = "BuGn") + 
    ylab("")
Surage = ggplot(train, aes(x = Age, fill = Survived)) + geom_histogram() + scale_fill_brewer(palette = "BuGn")
SurFam = ggplot(train, aes(x = SibSp + Parch, fill = Survived)) + geom_histogram() + 
    scale_fill_brewer(palette = "BuGn") + ylab("")
SurFare = ggplot(train, aes(x = Fare, fill = Survived)) + geom_histogram() + 
    scale_fill_brewer(palette = "BuGn") + ylab("")
SurSex = ggplot(train, aes(x = Sex, fill = Survived)) + geom_bar() + scale_fill_brewer(palette = "BuGn") + 
    ylab("")
SurEmbarked = ggplot(train, aes(x = Embarked, fill = Survived)) + geom_bar() + 
    scale_fill_brewer(palette = "BuGn") + ylab("")
grid.arrange(SurPc, Surage, SurFam, SurFare, SurSex, SurEmbarked, ncol = 3)
```

![](titanic-2_files/figure-markdown_github/unnamed-chunk-8-1.png)

From the diagram, we have a guess: people who are women, young but not too young, have 1-2 companies have more chance to survive.
