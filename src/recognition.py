import argparse
import json
from typing import Tuple, List, Any

import cv2
import editdistance
from path import Path

from src.dataloader_dataset import Loader, Batch
from src.model import Model
from src.preprocessor import Preprocessor


class FilePaths:
    fn_char_list = '/home/jobkuzin/study/ml/TextRecognition/model/charList.txt'
    fn_summary = '/home/jobkuzin/study/ml/TextRecognition/model/summary.json'
    fn_corpus = '../dataset/corpus.txt'


def get_img_height() -> int:
    return 32


def get_img_size() -> Tuple[int, int]:
    return 256, get_img_height()


def write_summary(average_train_loss: List[float], char_error_rates: List[float], word_accuracies: List[float]) -> None:
    with open(FilePaths.fn_summary, 'w') as f:
        json.dump({'averageTrainLoss': average_train_loss, 'charErrorRates': char_error_rates,
                   'wordAccuracies': word_accuracies}, f)


def char_list_from_file() -> List[str]:
    with open(FilePaths.fn_char_list) as f:
        return list(f.read())


def train(model: Model,
          loader: Loader,
          early_stopping: int = 25) -> None:
    epoch = 0
    summary_char_error_rates = []
    summary_word_accuracies = []

    train_loss_in_epoch = []
    average_train_loss = []

    preprocessor = Preprocessor(get_img_size())
    best_char_error_rate = float('inf')
    no_improvement_since = 0

    while True:
        epoch += 1
        print('Epoch:', epoch)

        # train
        print('Train NN')
        loader.train_set()
        while loader.has_next():
            iter_info = loader.get_iterator_info()
            batch = loader.get_next()
            batch = preprocessor.process_batch(batch)
            loss = model.train_batch(batch)
            print(f'Epoch: {epoch} Batch: {iter_info[0]}/{iter_info[1]} Loss: {loss}')
            train_loss_in_epoch.append(loss)

        char_error_rate, word_accuracy = validate(model, loader)

        summary_char_error_rates.append(char_error_rate)
        summary_word_accuracies.append(word_accuracy)
        average_train_loss.append((sum(train_loss_in_epoch)) / len(train_loss_in_epoch))
        write_summary(average_train_loss, summary_char_error_rates, summary_word_accuracies)

        train_loss_in_epoch = []

        if char_error_rate < best_char_error_rate:
            print('Character error rate improved, save model')
            best_char_error_rate = char_error_rate
            no_improvement_since = 0
            model.save()
        else:
            print(f'Character error rate not improved, best so far: {best_char_error_rate * 100.0}%')
            no_improvement_since += 1

        if no_improvement_since >= early_stopping:
            print(f'No more improvement for {early_stopping} epochs. Training stopped.')
            break


def validate(model: Model, loader: Loader) -> Tuple[float, float]:
    print('Validate NN')
    loader.validation_set()
    preprocessor = Preprocessor(get_img_size())
    num_char_err = 0
    num_char_total = 0
    num_word_ok = 0
    num_word_total = 0
    while loader.has_next():
        iter_info = loader.get_iterator_info()
        print(f'Batch: {iter_info[0]} / {iter_info[1]}')
        batch = loader.get_next()
        batch = preprocessor.process_batch(batch)
        recognized, _ = model.infer_batch(batch)

        print('Ground truth -> Recognized')
        for i in range(len(recognized)):
            num_word_ok += 1 if batch.gt_texts[i] == recognized[i] else 0
            num_word_total += 1
            dist = editdistance.eval(recognized[i], batch.gt_texts[i])
            num_char_err += dist
            num_char_total += len(batch.gt_texts[i])
            print('[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + batch.gt_texts[i] + '"', '->',
                  '"' + recognized[i] + '"')

    char_error_rate = num_char_err / num_char_total
    word_accuracy = num_word_ok / num_word_total
    print(f'Character error rate: {char_error_rate * 100.0}%. Word accuracy: {word_accuracy * 100.0}%.')
    return char_error_rate, word_accuracy


def infer(model: Model, fn_img: Path) -> None:
    img = cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)
    assert img is not None

    preprocessor = Preprocessor(get_img_size(), dynamic_width=True, padding=16)
    img = preprocessor.process_img(img)

    batch = Batch([img], None, 1)
    recognized, probability = model.infer_batch(batch, True)
    print(f"Recognized='{recognized[0]}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', choices=['train', 'validate', 'infer'], default='infer')
    parser.add_argument('--batch_size', help='Batch size.', type=int, default=100)
    parser.add_argument('--data_dir', help='Directory containing IAM dataset.', type=Path, required=False)
    parser.add_argument('--img_file', help='Image used for inference.', type=Path, default='../dataset/test.jpg')
    parser.add_argument('--early_stopping', help='Early stopping epochs.', type=int, default=25)

    return parser.parse_args()


def recognize(image_path: Path) -> tuple[Any, Any]:
    model = Model(char_list_from_file())
    recognized, probability = infer(model, image_path)
    return recognized, probability


def main() -> None:
    args = parse_args()

    # train the model
    if args.mode == 'train':
        loader = Loader(args.data_dir, args.batch_size, fast=args.fast)

        char_list = loader.char_list
        if args.line_mode and ' ' not in char_list:
            char_list = [' '] + char_list

        with open(FilePaths.fn_char_list, 'w') as f:
            f.write(''.join(char_list))

        with open(FilePaths.fn_corpus, 'w') as f:
            f.write(' '.join(loader.train_words + loader.validation_words))

        model = Model(char_list)
        train(model, loader, early_stopping=args.early_stopping)

    elif args.mode == 'validate':
        loader = Loader(args.data_dir, args.batch_size, fast=args.fast)
        model = Model(char_list_from_file())
        validate(model, loader, )

    elif args.mode == 'infer':
        model = Model(char_list_from_file())
        print("model")
        infer(model, args.img_file)


if __name__ == '__main__':
    main()
