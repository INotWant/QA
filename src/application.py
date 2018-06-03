# -*- coding: utf-8 -*-

import logging
import sys

import config
import network
import paddle.v2 as paddle
import reader
import utils

logger = logging.getLogger("paddle")
logger.setLevel(logging.ERROR)


class Application(object):
    def __init__(self, conf):
        self.conf = conf
        self.settings = reader.Settings(
            vocab=conf.vocab, is_training=False, label_schema=conf.label_schema)

        # init paddle
        # TODO(lipeng17) v2 API does not support parallel_nn yet. Therefore, we
        # can only use CPU currently
        paddle.init(use_gpu=conf.use_gpu, trainer_count=conf.trainer_count)

        # define network
        self.tags_layer = network.inference_net(conf)

    def infer(self, model_path, data_path):

        # load the trained models
        parameters = paddle.parameters.Parameters.from_tar(
            utils.open_file(model_path, "r"))
        inferer = paddle.inference.Inference(
            output_layer=self.tags_layer, parameters=parameters)

        # get question and evidences
        content = []
        f = open(data_path)
        for line in f.readlines():
            content.append(line.strip())
        question = []
        for x in content[0].split(' '):
            question.append(x)
        evidences = []
        for x in content[1:4]:
            e = []
            for y in x.split(' '):
                e.append(y)
            evidences.append(e)
        test_batch = []
        i = 0
        for evidence in evidences:
            test_batch.append(
                self.application_reader(question, evidence, self.get_qe(question, evidence), self.get_ee(i, evidences)))
            i += 1

        def count_evi_ids(test_batch):
            num = 0
            for sample in test_batch:
                num += len(sample[reader.E_IDS])
            return num

        tags = inferer.infer(
            input=test_batch, field=["id"], feeding=network.feeding)
        evi_ids_num = count_evi_ids(test_batch)
        assert len(tags) == evi_ids_num
        print(";\n".join(str(tag) for tag in tags) + ";")

    def application_reader(self, question, evidence, qe, ee):

        def get_unicode(collection):
            result = []
            for x in collection:
                result.append(unicode(x, 'utf-8'))
            return result

        question_unicode = get_unicode(question)
        evidence_unicode = get_unicode(evidence)

        result = []
        result.append([self.settings.vocab.get(token, self.settings.oov_id) \
                       for token in question_unicode])
        result.append([self.settings.vocab.get(token, self.settings.oov_id) \
                       for token in evidence_unicode])
        result.append([2 for _ in range(len(evidence))])
        result.append(qe)
        result.append(ee)
        return result

    def get_qe(self, question, evidence):
        result = []
        d = dict()
        for x in question:
            d[x] = 1
        for x in evidence:
            if d.get(x) is not None:
                result.append(1)
            else:
                result.append(0)
        return result

    def get_ee(self, pos, evidences):
        result = []
        d = dict()
        evidence = evidences[pos]
        i = 0
        for _ in evidences:
            if i != pos:
                for y in evidences[i]:
                    d[y] = 1
            i += 1
        for x in evidence:
            if d.get(x) is not None:
                result.append(1)
            else:
                result.append(0)
        return result


def main():
    # start_time = time()
    conf = config.InferConfig()
    conf.vocab = utils.load_dict(conf.word_dict_path)
    logger.info("length of word dictionary is : %d." % len(conf.vocab))

    application = Application(conf)
    application.infer('/home/QA/models/params_pass_00024.tar.gz', sys.argv[1])
    # end_time = time()
    # print("SpendTime :: ", (end_time - start_time))


if __name__ == '__main__':
    main()
