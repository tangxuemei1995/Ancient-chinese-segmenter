# （AnChinseseSeg）古汉语分词及词性标注工具 Word Segmentation and Part-of-speech for Ancient Chinese
基于2022年的分词文章，做了古汉语的分词和词性标注

这是一个非常粗糙朴素的分词和标注词性的工具

词性效果评估如下：

P:	92.82

R:	92.85

F:	92.84

分词效果评估如下：

P:	97.19

R:	97.22

F:	97.20

## Citation
词性标注并没有发表论文，但是如果您使用了我们的工具进行了学术研究，可以引用以下论文，我们是在该论文的基础上实现的
```
@inproceedings{tang-su-2022-slepen,
    title = "That Slepen Al the Nyght with Open Ye! Cross-era Sequence Segmentation with Switch-memory",
    author = "Tang, Xuemei  and
      Su, Qi",
    booktitle = "Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.acl-long.540",
    doi = "10.18653/v1/2022.acl-long.540",
    pages = "7830--7840",
}
```

## Requirements
环境配置请查看 requriments.txt

## How to use it
1)请从百度网盘或者google drive下载模型model.dt放到model文件夹中

baidu链接: https://pan.baidu.com/s/1jIbqk5b4GYBEMAdBPVJwYg 提取码: dac4

google drive: https://drive.google.com/drive/folders/1zFK30h6PQYRDDZ2uEScLy0l5VoC7jXHU?usp=sharing

2)将数据转成需要的格式

在data文件夹中有转化格式的py文件，需要修改data.py中的中的文件路径

```
for line in open('./zztj/1.txt')
```

然后运行data.py文件，即可完成文件预处理(包含繁简转化以及切句，切句是因为我们的模型对输入的句子长度有限制，每条数据不超过160字)

```
#python data.py
```

生成的test.txt是用于分词的文件

3）退出data文件夹，回到segmenter之下

#bash seg.sh
 
最后分好词的文件会model/之下test_result.txt，格式如下：
```
端明殿_NA 学士_NA 兼_VT 翰林侍读_NA 学士_NA 朝散大夫_NA 右谏议大夫_NA 充_VT 集贤院_NA
```
4）词性标记参考台湾中央研究院

https://lingcorpus.iis.sinica.edu.tw/kiwi/dkiwi/middle_chinese_c_wordtype.html

https://lingcorpus.iis.sinica.edu.tw/kiwi/akiwi/ancient_mandarin_chinese_c_wordtype.html

## Contact

Please contact us at tangxuemei@stu.pku.edu.cn if you have any questions.
Welcome to Research Center for Digital Humanities of Peking University! https://pkudh.org

