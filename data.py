import torchtext
import torchtext.data as data

class StanfordNLI(data.TabularDataset):

    dirname = 'snli_1.0'
    name = 'snli'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, root='.data',
               train='snli_1.0_train.jsonl', validation='snli_1.0_dev.jsonl',
               test='snli_1.0_test.jsonl'):

        path = cls.download(root)

        if parse_field is None:
            return super(StanfordNLI, cls).splits(
                path, root, train, validation, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(StanfordNLI, cls).splits(
            path, root, train, validation, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field)},
            filter_pred=lambda ex: ex.label != '-')


class StanfordNLI_test(data.TabularDataset):

    dirname = 'snli_1.0'
    name = 'snli'

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None, root='.data',
               test='snli_1.0_test.jsonl'):

        path = cls.download(root)

        if parse_field is None:
            return super(StanfordNLI_test, cls).splits(
                path, root, test,
                format='json', fields={'sentence1': ('premise', text_field),
                                       'sentence2': ('hypothesis', text_field),
                                       'gold_label': ('label', label_field)},
                filter_pred=lambda ex: ex.label != '-')
        return super(StanfordNLI_test, cls).splits(
            path, root, test,
            format='json', fields={'sentence1_binary_parse':
                                   [('premise', text_field),
                                    ('premise_transitions', parse_field)],
                                   'sentence2_binary_parse':
                                   [('hypothesis', text_field),
                                    ('hypothesis_transitions', parse_field)],
                                   'gold_label': ('label', label_field)},
            filter_pred=lambda ex: ex.label != '-')