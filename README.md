# judge_html

之前在趋势科研实习了5个月，期间做了一个任务是将网页进行归类，比如分为html，JavaScript或是VBScript。这个为另外一个识别恶意流量的项目做的一个功能，因为不同的网页类型，结构也不同，所以抽取特征的方式也不也一样。一开始公司用的是简单的匹配关键词的方法，效果很不好，所以当时的mentor要求我用机器学习的方式重新区分。这个项目的一个难点在于之前的样本没有标签，需要自己标注。虽然有一些工具如virus total可以根据sha1判断文件类型，但是只是对少量样本，而且经过检验会有误判的情况。项目的另外一个难点在于这三种文件联系紧密，html文件中可能会包含js和vbs，而js和vbs也可以通过document.write这种语句写入一段html的表达式。因为在非html文件中，vbs出现的次数很少，主要是js。所以一开始先将问题简化为区分html和非html的二分类问题。

我采取的方式是通过使用正则表达式匹配的方法，首先取出一部分特征明显的html和非html文档。如果文档以html的tag的形式开头，则分为html，如果整篇文档都不包含任何html的tag，则认为是非html文档，不符合这两项的都标记为uncertain。通过这种方法可以确定出一小部分置信度比较高的样本。然后以这部分样本作为训练集，提取特征，训练模型去预测uncertain那部分的样本。提取特征时首先将html，vbs，和js的所有的保留词作为特征，因为每个特征只有一个词，也就是称为unigram，因为一个词难以反映词和词之间的相互的顺序，所以加入bigram和trigram。具体做法是先用tfidf的方法选出前1000个词，这些词都是bigram或者是trigram，然后用树分裂的方法计算这1000个词的特征重要性，选出前300个。将这300个特征加入到之前的unigram的特征中。通过这小部分的训练集提取的特征去训练模型，然后预测uncertain部分的样本，根据预测评分，选取置信度比较高的样本合并到训练集中，形成新的训练集，将置信度较低的样本归为新的uncertain的样本。
