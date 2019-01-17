# Bert_ChinesewordSegment
基于bert的中文分词模型，人民日报语料下F1-Score 97%

首先，git clone https://github.com/google-research/bert.git

将下载的代码 modeling.py、optimization.py、tokenization.py放到本工程目录下，结构如下：

    Bert_ChinesewordSegment
    
    |____ PEOPLEdata
    |____ output
    |____ modeling.py
    |____ optimization.py
    |____ tokenization.py
    |____ run_cut.py
    |____ evaluation.py

然后下载bert的中文预训练模型

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)

设置好预训练模型路径和数据源路径：$BERT_CHINESE_DIR、$PEOPLEcut

运行下面语句：
```
python3 run_cut.py   --task_name="people"   --do_train=True   --do_predict=True  --data_dir=$PEOPLEcut    --vocab_file=$BERT_CHINESE_DIR/vocab.txt   --bert_config_file=$BERT_CHINESE_DIR/bert_config.json   --init_checkpoint=$BERT_CHINESE_DIR/bert_model.ckpt    --max_seq_length=128    --train_batch_size=32    --learning_rate=2e-5   --num_train_epochs=3.0    --output_dir=./output/result_cut/
```

GPU环境下，运行3个epochs大概耗时28分钟

运行完毕之后会显示评估结果：
```
INFO:tensorflow:***** Eval results *****
INFO:tensorflow:  count = 9925
INFO:tensorflow:  precision_avg = 0.9794
INFO:tensorflow:  recall_avg = 0.9780
INFO:tensorflow:  f1_avg = 0.9783
INFO:tensorflow:  error_avg = 0.0213
```

测试数据生成的分词效果可以在./output/result_cut/seg_result.txt里面看到

代码解析 [简书:Bert系列（五）——中文分词实践...](https://www.jianshu.com/p/be0a951445f4)
