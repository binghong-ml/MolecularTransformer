import os
import argparse
from onmt.translate.translator import build_translator
import onmt.opts

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    import re
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return ' '.join(tokens)

class ReactionPredictor:

    def __init__(self, topk=2, gpu=0):
        model = os.path.join(os.path.dirname(__file__), 'STEREO_separated_augm_model_average_20.pt')

        parser = argparse.ArgumentParser(
            description='molecular transformer',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        onmt.opts.add_md_help_argument(parser)
        onmt.opts.translate_opts(parser)

        self.opt = parser.parse_args([
            '-batch_size', '10',
            '-max_length', '200',
            '-replace_unk',
            '-fast',
            '-src', '',
            '-model', model,
            '-beam_size', str(topk),
            '-n_best', str(topk),
            '-gpu', str(gpu)
        ])
        self.translator = build_translator(self.opt, report_score=True)

    def predict(self, reactant_list):
        tokenized_reactant_list = [smi_tokenizer(smi + '>') for smi in reactant_list]
        all_scores, all_predictions = self.translator.translate(
            src_data_iter=tokenized_reactant_list,
            batch_size=self.opt.batch_size,
            attn_debug=self.opt.attn_debug
        )

        product_list = []
        for tokenized_products in all_predictions:
            product_list.append([''.join(t.strip().split(' ')) for t in tokenized_products])

        return product_list

if __name__ == '__main__':
    rp = ReactionPredictor(topk=3, gpu=0)

    reactant_list = ['CS(=O)(=O)Cl.OCCCBr', 'Nc1ccc(C(=O)O)cc1.O=C(Cl)C1CCC1']
    product_list = rp.predict(reactant_list)

    print(product_list)