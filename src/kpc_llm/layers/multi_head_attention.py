'''
因果注意力，和dropout的基础上实现多头注意力
1 原本可以通过简单的多个 因果注意力的context_vector的叠加来形成多头注意力的逻辑
2 但是应为W_q ,k ,v的升维矩阵在数学本质上是和multi head 同构的，因为 W_qkv的out维度就是最终的输出维度。
3 所以只要把w_qkv的维度设置成设计的 context_vector_dim * num_heads 就可以了。最后的W_q,k,v的out_d 就是 unroll 展开后的 c_v_d * num_haeds
4 当然，所作的这一切的多头的目的是每一个头是一个特征的变化空间，最终这些多个头所戴白的不同的特征空间，过滤通道，或者说是特征变化空间，
最终需要一个 combine变换，也就是一个更抽象的变换空间来组合降维，以便最终把特征转化成低标签空间。这时候需要一个新的 out空间，可以用 nn.Linear来初始化。
'''
from torch import nn,Tensor,softmax,manual_seed
from kpc_llm.data_process.causal_attention_process import setUpNegativeInfMask
from kpc_llm.data_process.drop_out import add_drop_out
from kpc_llm.utils.logger import getlogger

logger = getlogger()
class Multi_Head_Attention(nn.Module):
    def __init__(self, in_d:int , num_head:int , single_head_d:int , qkv_bias=False ):
        super().__init__()
        self.num_head = num_head
        #真正的token_d单独一个token的embedding维度
        self.single_head_d = single_head_d
        self.in_d = in_d
        self.out_d = num_head * single_head_d
        # nn.Linear(列，行) =  A的transpose(AT) ,因为 out = A * x_input + bias 所以也是  out = x_input * AT + bias 所以 行 才是 nn.Linear的out_dim
        # pytorch nn.Linear 采用的是  out = x_input * AT + bias 
        # nn.linear 是一个 affine linear 仿射线性变换， 相仿线性变换。做了偏移的线性变换，严格线性变换不能移动0点
        self.W_q = nn.Linear(in_d,self.out_d,qkv_bias)
        self.W_k = nn.Linear(in_d,self.out_d,qkv_bias)
        self.W_v = nn.Linear(in_d,self.out_d,qkv_bias)
        #输入的维度暂时设置跟输出的 out_d一样，这里是combine 组合多个head的维度，自然是out_d
        self.W_com_multihead = nn.Linear(self.out_d,self.out_d,qkv_bias)

    def forward(self,input_batch):
        '''
        这里的input_batch应该是个批量的input所以这里的input_batch的shape是[batch,input_out_d,in_d]
        in_d和初始化函数定义的in_d一样
        '''
        #input_token_dim必须跟 self.in_d一样才可以保证 nn.Linear输出是out_d
        num_batch,num_token,input_token_dim = input_batch.shape

        #输出后样本维度是out_d一批的样本数是num_token了，token的维度变成了新的out_d的维度。特征空间变了
        queries = self.W_q(input_batch)
        keys = self.W_k(input_batch)
        values = self.W_v(input_batch)

        # unroll 展开最后的out_d维度为多头和单头维度的乘积，Tensorflow框架用reshap ，这里用view
        # 特征空间 又被分成了多个组，一个头就是一组特征空间维度，组数就是 所谓头数。 
        # 不同的组分别从各自的空间中去路径寻优，因为现实中一句话的续写可以有多种方式，叫同义句，这个多头空间对应的是这个特征的捕捉。
        queries = queries.view(num_batch,num_token,self.num_head,self.single_head_d)
        keys = keys.view(num_batch,num_token,self.num_head,self.single_head_d)
        values = values.view(num_batch,num_token,self.num_head,self.single_head_d)

        """ 
        #这里要把 token数量 和 头数交换位置 ：原因如下
        多头注意力机制（Multi-Head Attention）的精妙所在。
        比如：XX.transpose(1,2)
        交换前 (32, 50, 8, 64)：意思是，这句话有 50 个词，每个词里面拆出了 8 个头。此时，不同的“头”被塞在每个词的内部。
        交换后 (32, 8, 50, 64)：意思是，我把这 8 个头彻底剥离出来。现在看起来，就像是有 8 个独立的平行世界（8 个 Head），每个世界里都有一句包含了 50 个词、每个词维度是 64 的句子。
        这样交换之后，PyTorch 就可以让这 8 个头并行去计算各自的 Attention 矩阵，互不干扰。
        """
        queries.transpose(1,2)
        keys.transpose(1,2)
        values.transpose(1,2)

        #计算 attention_scores 因为是超过2维的矩阵，不能用.T来转置
        logger.info(f'keys shape : {keys.shape}')
        attention_scores = queries @ keys.transpose(2,3)
        #添加因果注意力，防止未来词元被窥探影响训练
        attention_causal_scores = setUpNegativeInfMask(attention_scores,self)
        #scale得到归一化的分数，权重，
        attention_causal_weights = softmax(attention_causal_scores / queries.shape[-1]**0.5 ,dim = -1)
        #添加dropout随机关闭部分神经元
        attention_causal_drop_weights = add_drop_out(attention_causal_weights,0.5)
        #计算得到context_vector,
        context_vector = attention_causal_drop_weights @ values
        #目前上下文向量最后两个维度本质上是 num_head * head_d = out_d所以可以转置1,2后com合并最后连个维度
        con_vector_com =  context_vector.transpose(1,2).contiguous().view(num_batch,num_token,self.out_d)
        #W_com_multihead 也是要一层新的组合特征用的层，(可选层)
        com_heads_c_vecters =  self.W_com_multihead(con_vector_com)
        return com_heads_c_vecters

if __name__ == '__main__':
    manual_seed(517)
    #定义两个批次的数据
    inputs = Tensor([
        [[0.43, 0.15, 0.89], # Your 
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one 
        [0.05, 0.80, 0.55]] # step
        ,
         [[0.43, 0.15, 0.89], # Your 
        [0.55, 0.87, 0.66], # journey
        [0.57, 0.85, 0.64], # starts
        [0.22, 0.58, 0.33], # with 
        [0.77, 0.25, 0.10], # one 
        [0.05, 0.80, 0.55]] # step
    ])
    logger.info(f'input shape : {inputs.shape}')
    m_attention = Multi_Head_Attention(3,3,3)
    com_heads_c_vecters = m_attention(inputs)
    #should be 2,6,9
    logger.info(f'com_heads_c_vecters : {com_heads_c_vecters}')
    logger.info(f'shape : {com_heads_c_vecters.shape}')