import torch
import argparse
import sys
from machamp.model import trainer
from machamp.predictor.predict import predict_with_paths
from transformers import RobertaTokenizer, RobertaModel, RobertaForSequenceClassification


tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)

print(output)

# https://github.com/machamp-nlp/machamp

# Training-----------------------------------------------------------------#

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_configs", nargs='+',
                    help="Path(s) to dataset configurations (use --sequential to train on them sequentially, "
                         "default is joint training).")
parser.add_argument("--name", default="", type=str, help="Log dir name.")
parser.add_argument("--sequential", action="store_true",
                    help="Enables finetuning sequentially, this will train the same weights once for each "
                         "dataset_config you pass.")
parser.add_argument("--parameters_config", default="configs/params.json", type=str,
                    help="Configuration file for parameters of the model.")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU.")
model_dir_group = parser.add_mutually_exclusive_group()
model_dir_group.add_argument("--resume", default='', type=str,
                             help='Finalize training on a model for which training abruptly stopped. Give the path to the log '
                                  'directory of the model.')
model_dir_group.add_argument("--model_dir", default=None, type=str,
                             help='Specify a directory to store model and logs in. Overrides the default.')
parser.add_argument("--retrain", type=str, default='',
                    help="Retrain on an previously train MaChAmp model. Specify the path to model.tar.gz and add a "
                         "dataset_config that specifies the new training.")
parser.add_argument("--seed", type=int, default=8446, help="seed to use for training.")
args = parser.parse_args()

if args.resume == '' and (args.dataset_configs == None or len(args.dataset_configs) == 0):
    print('Please provide at least 1 dataset configuration')
    exit(1)

if args.device == None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.device)

name = args.name
if args.resume == '' and name == '':
    names = [name[name.rfind('/') + 1: name.rfind('.') if '.' in name else len(name)] for name in args.dataset_configs]
    name = '.'.join(names)
if args.resume != '':
    name = args.resume.split('/')[1]

cmd = ' '.join(sys.argv)

if args.sequential:
    if args.model_dir:
        prevDir = trainer.train(name, args.parameters_config, [args.dataset_configs[0]], device, args.resume, args.retrain,
                                args.seed, cmd, args.model_dir + '/0')
        for datasetIdx, dataset in enumerate(args.dataset_configs[1:]):
            prevDir = trainer.train(name, args.parameters_config, [dataset], device, None, prevDir, args.seed, cmd, args.model_dir + f'/{datasetIdx + 1}')
    else:
        prevDir = trainer.train(name + '/0', args.parameters_config, [args.dataset_configs[0]], device, args.resume, args.retrain,
                                args.seed, cmd)
        for datasetIdx, dataset in enumerate(args.dataset_configs[1:]):
            modelName = name + '/' + str(datasetIdx + 1)
            prevDir = trainer.train(modelName, args.parameters_config, [dataset], device, None, prevDir, args.seed, cmd)
else:
    trainer.train(name, args.parameters_config, args.dataset_configs, device, args.resume, args.retrain, args.seed, cmd, args.model_dir)



# Prediction-----------------------------------------------------------------#

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
#                     level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
# logger = logging.getLogger(__name__)

# parser = argparse.ArgumentParser()
# parser.add_argument("torch_model", type=str, help="The path to the pytorch (*.pt) model.")
# parser.add_argument("file_paths", nargs='+',
#                     help="contains a list of input and output files. You can predict on multiple files by having a "
#                          "structure like: input1 output1 input2 output2.")
# parser.add_argument("--dataset", default=None, type=str,
#                     help="name of the dataset, needed to know the word_idx/sent_idxs to read from")
# parser.add_argument("--device", default=None, type=int, help="CUDA device number; set to -1 for CPU.")
# parser.add_argument("--batch_size", default=32, type=int, help="The size of each prediction batch.")
# parser.add_argument("--raw_text", action="store_true", help="Input raw sentences, one per line in the input file.")
# parser.add_argument("--topn", default=None, type=int, help='Output the top-n labels and their probability.')
# parser.add_argument("--conn", default='=', type=str, help="With --topn, string inserted between each label and its probability.")
# parser.add_argument("--sep", default='|', type=str, help="With --topn, string inserted between label-probability pairs.")
# parser.add_argument("--threshold", default=None, type=float, help="The threshold to be used for multiseq and multiclas, note that the same metric will be applied to all tasks.")
# parser.add_argument("--max_sents", default=-1, type=int, help="The maximum number of sentences (i.e. lines) to process.")
# args = parser.parse_args()

# logger.info('cmd: ' + ' '.join(sys.argv) + '\n')
# if len(args.file_paths) % 2 == 1:
#     logger.error(
#         'Error: the number of files passed is not even. You need to pass an output file for each input file: ' + str(
#             args.file_paths))
#     exit(1)

# if args.device == None:
#     device = "cuda:0" if torch.cuda.is_available() else "cpu"
# elif args.device == -1:
#     device = 'cpu'
# else:
#     device = 'cuda:' + str(args.device)

# logger.info('loading model...')
# model = torch.load(args.torch_model, map_location=device, weights_only=False)
# model.device = device

# if args.topn != None:
#     for decoder in model.decoders:
#         model.decoders[decoder].topn = args.topn

# for dataIdx in range(0, len(args.file_paths), 2):
#     input_path = args.file_paths[dataIdx]
#     output_path = args.file_paths[dataIdx + 1]
#     logger.info('predicting on ' + input_path + ', saving on ' + output_path)
#     predict_with_paths(model, input_path, output_path, args.dataset, args.batch_size, args.raw_text, device, args.conn, args.sep, args.threshold, args.max_sents)