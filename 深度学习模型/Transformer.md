## Encoder
1) Embed(src)
2) $MultiHead(X,X,X)$
3) $LayerNorm(X)$
4) Feedforward+ResidualAdd
5) LayerNorm(X)  
注意：  
1) MultiHeadAttention中W的列数为embed_size//num_heads  
2) 因为句子中会有填充符号，比如(P补长，)，需要mask来遮盖住这个词，让attention不注意它。
## Decoder
1) encdec_attention  
Q=target  
K=V=src (enc_out)

## 预备
### Attention机制 
Q.shape=K.shape=V.shape=(n,k)  
Attention.shape=(n,k)@(k,n)@(n,k)=(n,k)

$$
Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V
$$

- Self-Attention

$$
self-Attention(X)=Attention(XW^Q,XW^K,XW^V)
$$

- MultiHeadAttention  
参数:
1) num_heads,头的个数
2) 单个头注意的单词个数（W的列数，生成的特征个数）

$$
\begin{align}
MultiHead(Q,K,V)&=Concat(head_1,\cdots,head_h)W^O\\
where\ head_i&=Attention(QW_i^Q,KW_i^K,VW_i^V)
\end{align}
$$

## Bert
只包含Eecoders，文本分类等
## GPT
只包含Decoders，生成文本等

## Transformer的问题
基于Transformer的模型已经被证明了在许多NLP任务中的价值，但这类模型的时间复杂度、内存使用复杂度都是 n2 （n为序列长度）。  
`Longformer`, `Performer`, `Reformer`, `Clustered attention`都试图通过近似全主力机制改善该问题。  
`BigBird`[论文](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2007.14062)是处理这类问题最新模型的其中之一，它使用`block sparse attention`替换了原类似Bert一样的全注意力机制，在与BERT一样的计算力情况下，可以处理的序列长度达到4096。它已经在很多长文本序列的任务上达到SOTA效果，例如长文本摘要、长文本问答。 BigBird RoBERTa模型现在已经可以在Transformers仓库中使用。
1) [BigBird](https://zhuanlan.zhihu.com/p/444333724)
2) [使用Transformer改进TextCnn](https://www.infoq.cn/article/7iidpz0v4mt43b9ruit3)