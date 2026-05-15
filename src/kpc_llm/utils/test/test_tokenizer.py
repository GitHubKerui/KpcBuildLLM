from kpc_llm.utils.tokenizer import Tokenizer
from kpc_llm.utils.logger import getlogger

logger = getlogger()
def test_getPreProcessed():
    """
    测试getPreProcessed
    """
    try:
        words = Tokenizer.getPreProcessed('the-verdict.txt','data')
        logger.info(f'处理后的字符串长度 : {len(words)}')
        logger.info(f'测试看看处理后的前10个字符串 : {words[:10]}')
        logger.info(f'测试成功')
    except Exception as e :
        logger.debug(f'出现错误 : {e}',exc_info=True)

def test_encode_decode():
    """
    测试getPreProcessed
    """
    try:
        tokenizer = Tokenizer('the-verdict.txt','data')
        tokenIds = tokenizer.encode()
        tokens = tokenizer.decode(tokenIds)
        outTxt = tokenizer.decode2Txt(tokenIds)
        logger.info(f'处理后的tokenIds长度 : {len(tokenIds)}')
        logger.info(f'测试看看处理后的前10个ids : {tokenIds[:10]}')
        logger.info(f'处理后的tokens长度 : {len(tokens)}')
        logger.info(f'测试看看处理后的前10个tokens : {tokens[:10]}')
        logger.info(f'测试看看处理后的文本 : {outTxt[:100]}')
        logger.info(f'测试成功')
    except Exception as e :
        logger.debug(f'出现错误 : {e}',exc_info=True)

def test_unk_endoftext():
    try:
        tokenizer = Tokenizer('the-verdict.txt','data')
        tokenIds = tokenizer.encodeFromStr("you are kpc 哈哈哈 <|endoftext|> next article.")
        tokens = tokenizer.decode(tokenIds)
        tokensstr = tokenizer.decode2Txt(tokenIds)
        logger.info(f'测试test_unk_endoftext : {list(Tokenizer.VOCAB.items())[-5:]}')
        logger.info(f'测试<|unk|> : {Tokenizer.VOCAB.get('<|unk|>')}')
        logger.info(f'tokensstr : {tokensstr}')
        logger.info(f'测试成功')
    except Exception as e :
        logger.debug(f'出现错误 : {e}',exc_info=True)


if __name__ == "__main__":
   test_unit = ["test_unk_endoftext"]
   for case in test_unit:
        match case:
            case "test_getPreProcessed":
                test_getPreProcessed()
            case "test_encode_decode":
                test_encode_decode()
            case "test_unk_endoftext":
                test_unk_endoftext()